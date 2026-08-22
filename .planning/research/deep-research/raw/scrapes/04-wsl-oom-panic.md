[Skip to content](https://github.com/microsoft/WSL/issues/40732#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/microsoft/WSL/issues/40732) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/microsoft/WSL/issues/40732) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/microsoft/WSL/issues/40732) to refresh your session.Dismiss alert

{{ message }}

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/microsoft/WSL/issues/40732).

[microsoft](https://github.com/microsoft)/ **[WSL](https://github.com/microsoft/WSL)** Public

- [Notifications](https://github.com/login?return_to=%2Fmicrosoft%2FWSL) You must be signed in to change notification settings
- [Fork\\
1.8k](https://github.com/login?return_to=%2Fmicrosoft%2FWSL)
- [Star\\
33.5k](https://github.com/login?return_to=%2Fmicrosoft%2FWSL)


# ⚠️ WSL2 GPU OOM leads to Hyper-V kernel panic and Windows BLUE SCREEN OF DEATH — total system crash, not a recoverable error\#40732

New issue

Copy link

New issue

Copy link

Open

Open

[⚠️ WSL2 GPU OOM leads to Hyper-V kernel panic and Windows BLUE SCREEN OF DEATH — total system crash, not a recoverable error](https://github.com/microsoft/WSL/issues/40732#top)#40732

Copy link

Labels

[GPU](https://github.com/microsoft/WSL/issues?q=state%3Aopen%20label%3A%22GPU%22)

## Description

[![@wzgrx](https://avatars.githubusercontent.com/u/39661556?v=4&size=48)](https://github.com/wzgrx)

[wzgrx](https://github.com/wzgrx)

opened [on Jun 6on Jun 6, 2026](https://github.com/microsoft/WSL/issues/40732#issue-4602776075)

Issue body actions

## Bug Description

WSL2 GPU OOM (Out of Memory) due to repeated CUDA allocation failures leads to a **Hyper-V VM kernel panic** followed by a **Windows host bugcheck (BSOD)** — the entire system crashes and reboots, not just a recoverable CUDA error.

## Steps to Reproduce

1. Load a large model (>18GB VRAM) in WSL2 via PyTorch or llama.cpp — e.g., Qwen3.6-35B on an RTX 5090 (24GB)
2. While it is loaded, attempt to load a second model (e.g., LocateAnything-3B, ~14GB)
3. PyTorch CUDA caching allocator / CUDA driver API returns OOM
4. Retry without cleaning up VRAM fragments 3-5 times
5. ⚠️ **Instead of returning a clean CUDA error, the WSL2 VM kernel panics, and Windows host bugchecks (BSOD)**
6. Full system restart required

## Expected Behavior

CUDA OOM should return a clean error code (`cudaErrorMemoryAllocation`). The process should fail gracefully. The WSL VM should never crash the Windows host.

## Actual Behavior

Consecutive GPU OOM events corrupt the GPU-PV (GPU ParaVirtualization) translation layer state, causing:

1. Hyper-V VM kernel panic (WSL crashes internally)
2. Windows host unable to recover → bugcheck (BLUE SCREEN)
3. Full system reboot — all unsaved work lost

## Environment

- **Windows 11** 23H2
- **GPU:** NVIDIA GeForce RTX 5090 Laptop GPU (Blackwell sm\_120)
- **NVIDIA Driver:** 610.47
- **WSL2 Kernel:** 6.18.26.1-microsoft-standard-WSL2
- **CUDA:** 13.3
- **VRAM:** 24 GB total (only ~8 GB practically usable in WSL2 GPU-PV)

## Notes

- Single large models work fine in WSL2 via both PyTorch and llama.cpp
- The crash ONLY happens with **consecutive OOM failures** from repeated allocation attempts
- This is a **WSL2 GPU-PV layer stability issue** — the virtualized GPU driver should never be able to crash the entire Hyper-V VM and host

## Related

- [\[Bug\] WSL2 GPU passthrough has 16 GiB CUDA driver overhead on Blackwell sm\_120 — blocks PyTorch large allocations #40401](https://github.com/microsoft/WSL/issues/40401) (16GB GPU-PV driver overhead on Blackwell)
- [WSL 2 consumes massive amounts of RAM and doesn't return it #4166](https://github.com/microsoft/WSL/issues/4166) (WSL2 memory management)

## Impact

Data loss from unsaved work, productivity disruption, and erodes trust in WSL2 for GPU workloads. This makes WSL2 unusable for any development involving large GPU models where OOM is a risk.

## Activity

### github-actions commented on Jun 6on Jun 6, 2026

[![@github-actions](https://avatars.githubusercontent.com/in/15368?v=4&size=48)](https://github.com/apps/github-actions)

[github-actions](https://github.com/apps/github-actions) bot

[on Jun 6on Jun 6, 2026](https://github.com/microsoft/WSL/issues/40732#issuecomment-4638102811) – with [GitHub Actions](https://help.github.com/en/actions)

Contributor

More actions

# Logs are required for review from WSL team

If this a feature request, please reply with '/feature'. If this is a question, reply with '/question'.

**Otherwise, please attach logs by following the instructions below**, your issue will not be reviewed unless they are added. These logs will help us understand what is going on in your machine.

How to collect WSL logs

Download and execute [collect-wsl-logs.ps1](https://github.com/Microsoft/WSL/blob/master/diagnostics/collect-wsl-logs.ps1) in an **administrative powershell prompt**:

```
Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/microsoft/WSL/master/diagnostics/collect-wsl-logs.ps1" -OutFile collect-wsl-logs.ps1
Set-ExecutionPolicy Bypass -Scope Process -Force
.\collect-wsl-logs.ps1
```

The script will output the path of the log file once done.

If this is a networking issue, please use `.\collect-wsl-logs.ps1 -LogProfile networking` instead, following the instructions in [Collect WSL logs for networking issues](https://github.com/microsoft/WSL/blob/master/CONTRIBUTING.md#collect-wsl-logs-for-networking-issues)

Once completed please upload the output files to this GitHub issue.

See [Collect WSL logs (recommended method)](https://github.com/microsoft/WSL/blob/master/CONTRIBUTING.md#8-collect-wsl-logs-recommended-method).

If you choose to email these logs instead of attaching them to the bug, please send them to [wsl-gh-logs@microsoft.com](mailto:wsl-gh-logs@microsoft.com) with the GitHub issue number in the subject, and include a link to your GitHub issue comment in the message body, and reply with '/emailed-logs'.

[![](https://avatars.githubusercontent.com/in/15368?s=64&v=4)github-actions](https://github.com/apps/github-actions)

added

[needs-author-feedback](https://github.com/microsoft/WSL/issues?q=state%3Aopen%20label%3A%22needs-author-feedback%22)

[on Jun 6on Jun 6, 2026](https://github.com/microsoft/WSL/issues/40732#event-26417398309)

### wzgrx commented on Jun 6on Jun 6, 2026

[![@wzgrx](https://avatars.githubusercontent.com/u/39661556?v=4&size=48)](https://github.com/wzgrx)

[wzgrx](https://github.com/wzgrx)

[on Jun 6on Jun 6, 2026](https://github.com/microsoft/WSL/issues/40732#issuecomment-4638137494)

Author

More actions

> # Logs are required for review from WSL team
>
> If this a feature request, please reply with '/feature'. If this is a question, reply with '/question'. **Otherwise, please attach logs by following the instructions below**, your issue will not be reviewed unless they are added. These logs will help us understand what is going on in your machine.
>
> How to collect WSL logs

[01-wsl-dmesg.log](https://github.com/user-attachments/files/28662733/01-wsl-dmesg.log)

[02-system-info.log](https://github.com/user-attachments/files/28662730/02-system-info.log)

[03-gateway-errors.log](https://github.com/user-attachments/files/28662732/03-gateway-errors.log)

[04-vision-errors.log](https://github.com/user-attachments/files/28662729/04-vision-errors.log)

[05-crash-timeline.txt](https://github.com/user-attachments/files/28662731/05-crash-timeline.txt)

[![](https://avatars.githubusercontent.com/in/95686?s=64&v=4)microsoft-github-policy-service](https://github.com/apps/microsoft-github-policy-service)

removed

[needs-author-feedback](https://github.com/microsoft/WSL/issues?q=state%3Aopen%20label%3A%22needs-author-feedback%22)

[on Jun 6on Jun 6, 2026](https://github.com/microsoft/WSL/issues/40732#event-26417539312)

[![](https://avatars.githubusercontent.com/u/19360522?s=64&u=bb0576cea2a69ca9d1b25897e5088e471a7c5b77&v=4)Feng Wang (chemwolf6922)](https://github.com/chemwolf6922)

added

[GPU](https://github.com/microsoft/WSL/issues?q=state%3Aopen%20label%3A%22GPU%22)

[on Jun 7on Jun 7, 2026](https://github.com/microsoft/WSL/issues/40732#event-26451650890)

[![](https://avatars.githubusercontent.com/u/65133687?s=64&u=facb7d6b98e0b4aa0835c7e509c4aeb843951950&v=4)liquidsnakeblue](https://github.com/liquidsnakeblue)

added a commit that references this issue [on Jul 11on Jul 11, 2026](https://github.com/microsoft/WSL/issues/40732#event-27847969574)

[WSL VM-death fix: never set expandable\_segments; buffer hygiene + VRA…](https://github.com/liquidsnakeblue/retro-dreamer/commit/5f62fc1ca0c304fce665f4a6a87bf70a07e3bfc3)

...

[5f62fc1](https://github.com/liquidsnakeblue/retro-dreamer/commit/5f62fc1ca0c304fce665f4a6a87bf70a07e3bfc3)

[Sign up for free](https://github.com/signup?return_to=https://github.com/microsoft/WSL/issues/40732)**to join this conversation on GitHub.** Already have an account? [Sign in to comment](https://github.com/login?return_to=https://github.com/microsoft/WSL/issues/40732)

## Metadata

## Metadata

### Assignees

No one assigned

### Labels

[GPU](https://github.com/microsoft/WSL/issues?q=state%3Aopen%20label%3A%22GPU%22)

### Type

No type

### Projects

No projects

### Milestone

No milestone

### Relationships

None yet

### Development

No branches or pull requests

### Participants

[![@chemwolf6922](https://avatars.githubusercontent.com/u/19360522?s=64&u=bb0576cea2a69ca9d1b25897e5088e471a7c5b77&v=4)](https://github.com/chemwolf6922)[![@wzgrx](https://avatars.githubusercontent.com/u/39661556?s=64&v=4)](https://github.com/wzgrx)

## Issue actions

- ![](https://github.githubassets.com/assets/github-copilot-app-light-15ad5534265eeacd.svg)Open in GitHub Copilot app

You can’t perform that action at this time.