#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <clocale>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

struct NodeTiming {
    std::string op_name;
    std::string tensor_name;
    std::vector<int64_t> ne;
    std::string type;
    size_t nbytes;
    std::string phase; // "prefill" or "decode"
    int step;
    double duration_us;
};

struct ProfilerState {
    std::string current_phase = "prefill";
    int current_step = 0;
    std::chrono::high_resolution_clock::time_point node_start_time;
    std::vector<NodeTiming> timings;
    bool profiling_active = true;
};

static ProfilerState g_profiler;

static bool profiler_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (!g_profiler.profiling_active) {
        return true;
    }

    if (ask) {
        // Record start time before kernel launch
        g_profiler.node_start_time = std::chrono::high_resolution_clock::now();
        return true;
    }

    // ask == false: Node execution complete
    auto end_time = std::chrono::high_resolution_clock::now();
    double dur_us = std::chrono::duration<double, std::micro>(end_time - g_profiler.node_start_time).count();

    NodeTiming rec;
    rec.op_name = ggml_op_desc(t);
    rec.tensor_name = t->name ? t->name : "";
    rec.ne = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    rec.type = ggml_type_name(t->type);
    rec.nbytes = ggml_nbytes(t);
    rec.phase = g_profiler.current_phase;
    rec.step = g_profiler.current_step;
    rec.duration_us = dur_us;

    g_profiler.timings.push_back(rec);

    return true;
}

struct OpStats {
    std::string op_name;
    int count = 0;
    double total_us = 0.0;
    double min_us = 1e12;
    double max_us = 0.0;
};

static std::map<std::string, OpStats> aggregate_by_op(const std::vector<NodeTiming> & timings, const std::string & phase_filter = "") {
    std::map<std::string, OpStats> stats;
    for (const auto & rec : timings) {
        if (!phase_filter.empty() && rec.phase != phase_filter) {
            continue;
        }
        auto & entry = stats[rec.op_name];
        entry.op_name = rec.op_name;
        entry.count++;
        entry.total_us += rec.duration_us;
        if (rec.duration_us < entry.min_us) entry.min_us = rec.duration_us;
        if (rec.duration_us > entry.max_us) entry.max_us = rec.duration_us;
    }
    return stats;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    std::string out_json_path = "profile_out.json";
    int n_predict = 32;

    std::vector<char*> filtered_argv;
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--out-json" && i + 1 < argc) {
            out_json_path = argv[++i];
        } else if (arg == "-n" || arg == "--predict" || arg == "--n-predict") {
            if (i + 1 < argc) n_predict = std::stoi(argv[i + 1]);
            filtered_argv.push_back(argv[i]);
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    int filtered_argc = static_cast<int>(filtered_argv.size());
    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.cb_eval = profiler_cb_eval;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        std::cerr << "Error: failed to initialize llama context\n";
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> prompt_tokens;
    if (!params.prompt.empty()) {
        prompt_tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    } else {
        prompt_tokens = {llama_vocab_bos(vocab)};
    }

    std::cout << "=== Starting Profiler Run ===\n";
    std::cout << "Prompt tokens: " << prompt_tokens.size() << "\n";
    std::cout << "Gen tokens   : " << n_predict << "\n";

    // 1. Prefill Phase (Chunked by params.n_batch)
    g_profiler.current_phase = "prefill";
    g_profiler.current_step = 0;
    g_profiler.profiling_active = true;

    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    size_t batch_size = params.n_batch > 0 ? params.n_batch : 2048;
    for (size_t i = 0; i < prompt_tokens.size(); i += batch_size) {
        size_t n_eval = std::min<size_t>(prompt_tokens.size() - i, batch_size);
        if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data() + i, n_eval))) {
            std::cerr << "Error: failed during prefill eval chunk at " << i << "\n";
            return 1;
        }
    }
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_wall_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    // 2. Decode Phase (Greedy)
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    g_profiler.current_phase = "decode";

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    int actual_generated = 0;

    for (int i = 0; i < n_predict; ++i) {
        g_profiler.current_step = i + 1;

        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        actual_generated++;
        if (llama_decode(ctx, llama_batch_get_one(&new_token_id, 1))) {
            std::cerr << "Error: failed during decode step " << i << "\n";
            break;
        }
    }
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_wall_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

    llama_sampler_free(smpl);

    std::cout << "\nPrefill Wall Time: " << std::fixed << std::setprecision(2) << prefill_wall_ms << " ms ("
              << (prompt_tokens.size() / (prefill_wall_ms / 1000.0)) << " t/s)\n";
    std::cout << "Decode Wall Time : " << std::fixed << std::setprecision(2) << decode_wall_ms << " ms ("
              << (actual_generated / (decode_wall_ms / 1000.0)) << " t/s)\n";

    // Aggregations
    auto prefill_stats = aggregate_by_op(g_profiler.timings, "prefill");
    auto decode_stats = aggregate_by_op(g_profiler.timings, "decode");
    auto overall_stats = aggregate_by_op(g_profiler.timings, "");

    double total_prefill_gpu_us = 0.0;
    for (const auto & kv : prefill_stats) total_prefill_gpu_us += kv.second.total_us;

    double total_decode_gpu_us = 0.0;
    for (const auto & kv : decode_stats) total_decode_gpu_us += kv.second.total_us;

    double total_overall_gpu_us = total_prefill_gpu_us + total_decode_gpu_us;

    std::cout << "\n=== Top Prefill Operations (% GPU Time) ===\n";
    std::vector<OpStats> prefill_vec;
    for (const auto & kv : prefill_stats) prefill_vec.push_back(kv.second);
    std::sort(prefill_vec.begin(), prefill_vec.end(), [](const OpStats & a, const OpStats & b) {
        return a.total_us > b.total_us;
    });

    for (size_t i = 0; i < std::min<size_t>(10, prefill_vec.size()); ++i) {
        const auto & s = prefill_vec[i];
        double pct = (total_prefill_gpu_us > 0) ? (s.total_us / total_prefill_gpu_us * 100.0) : 0.0;
        std::cout << "  " << std::left << std::setw(20) << s.op_name
                  << " : " << std::right << std::setw(6) << std::fixed << std::setprecision(2) << pct << "% "
                  << "(" << std::setw(8) << std::fixed << std::setprecision(2) << (s.total_us / 1000.0) << " ms, "
                  << s.count << " calls, avg " << (s.total_us / s.count) << " us)\n";
    }

    std::cout << "\n=== Top Decode Operations (% GPU Time) ===\n";
    std::vector<OpStats> decode_vec;
    for (const auto & kv : decode_stats) decode_vec.push_back(kv.second);
    std::sort(decode_vec.begin(), decode_vec.end(), [](const OpStats & a, const OpStats & b) {
        return a.total_us > b.total_us;
    });

    for (size_t i = 0; i < std::min<size_t>(10, decode_vec.size()); ++i) {
        const auto & s = decode_vec[i];
        double pct = (total_decode_gpu_us > 0) ? (s.total_us / total_decode_gpu_us * 100.0) : 0.0;
        std::cout << "  " << std::left << std::setw(20) << s.op_name
                  << " : " << std::right << std::setw(6) << std::fixed << std::setprecision(2) << pct << "% "
                  << "(" << std::setw(8) << std::fixed << std::setprecision(2) << (s.total_us / 1000.0) << " ms, "
                  << s.count << " calls, avg " << (s.total_us / s.count) << " us)\n";
    }

    // Write JSON Report
    std::ofstream ofs(out_json_path);
    if (ofs.is_open()) {
        ofs << "{\n";
        ofs << "  \"prompt_tokens\": " << prompt_tokens.size() << ",\n";
        ofs << "  \"gen_tokens\": " << actual_generated << ",\n";
        ofs << "  \"prefill_wall_ms\": " << prefill_wall_ms << ",\n";
        ofs << "  \"decode_wall_ms\": " << decode_wall_ms << ",\n";
        ofs << "  \"total_prefill_gpu_us\": " << total_prefill_gpu_us << ",\n";
        ofs << "  \"total_decode_gpu_us\": " << total_decode_gpu_us << ",\n";
        ofs << "  \"total_overall_gpu_us\": " << total_overall_gpu_us << ",\n";

        // prefill_summary
        ofs << "  \"prefill_summary\": {\n";
        for (size_t i = 0; i < prefill_vec.size(); ++i) {
            const auto & s = prefill_vec[i];
            double pct = (total_prefill_gpu_us > 0) ? (s.total_us / total_prefill_gpu_us * 100.0) : 0.0;
            ofs << "    \"" << s.op_name << "\": {\"pct\": " << pct << ", \"total_us\": " << s.total_us
                << ", \"count\": " << s.count << ", \"avg_us\": " << (s.total_us / s.count) << "}";
            if (i + 1 < prefill_vec.size()) ofs << ",";
            ofs << "\n";
        }
        ofs << "  },\n";

        // decode_summary
        ofs << "  \"decode_summary\": {\n";
        for (size_t i = 0; i < decode_vec.size(); ++i) {
            const auto & s = decode_vec[i];
            double pct = (total_decode_gpu_us > 0) ? (s.total_us / total_decode_gpu_us * 100.0) : 0.0;
            ofs << "    \"" << s.op_name << "\": {\"pct\": " << pct << ", \"total_us\": " << s.total_us
                << ", \"count\": " << s.count << ", \"avg_us\": " << (s.total_us / s.count) << "}";
            if (i + 1 < decode_vec.size()) ofs << ",";
            ofs << "\n";
        }
        ofs << "  },\n";

        // overall_summary
        std::vector<OpStats> overall_vec;
        for (const auto & kv : overall_stats) overall_vec.push_back(kv.second);
        std::sort(overall_vec.begin(), overall_vec.end(), [](const OpStats & a, const OpStats & b) {
            return a.total_us > b.total_us;
        });

        ofs << "  \"overall_summary\": {\n";
        for (size_t i = 0; i < overall_vec.size(); ++i) {
            const auto & s = overall_vec[i];
            double pct = (total_overall_gpu_us > 0) ? (s.total_us / total_overall_gpu_us * 100.0) : 0.0;
            ofs << "    \"" << s.op_name << "\": {\"pct\": " << pct << ", \"total_us\": " << s.total_us
                << ", \"count\": " << s.count << ", \"avg_us\": " << (s.total_us / s.count) << "}";
            if (i + 1 < overall_vec.size()) ofs << ",";
            ofs << "\n";
        }
        ofs << "  }\n";
        ofs << "}\n";
        ofs.close();
        std::cout << "\nDetailed JSON report saved to: " << out_json_path << "\n";
    }

    llama_backend_free();
    return 0;
}
