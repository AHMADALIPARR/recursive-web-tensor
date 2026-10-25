# Self-hosted GPU runners (Jetson Orin NX / RTX 3080)

The `train-gpu` job runs on runners labeled `[self-hosted, gpu]`.
It runs ONLY on manual dispatch with the `gpu` input set to true.
Schedule runs never start it, and it is not on the publish path:
if no GPU runner is registered, the job stays queued and the CPU
chain still promotes.

## Registering a runner (on the hardware, not in this repo)

1. On GitHub: repository Settings -> Actions -> Runners -> New
   self-hosted runner. Pick Linux and your architecture (ARM64 for
   the Jetson Orin NX, x64 for the RTX 3080 box).
2. On the box, run the displayed `config.sh` command with the
   registration token GitHub shows you, adding the labels:
   `./config.sh --labels self-hosted,gpu`
3. Start the runner (`./run.sh`, or install the service with
   `./svc.sh install` + `./svc.sh start`).
4. Verify: the runner appears in Settings -> Actions -> Runners
   as Idle with the `gpu` label.

The registration token is entered on the box only. It must NEVER
be committed to this repo, and no workflow in this repo references
it. Workflows use `GITHUB_TOKEN` only.

## Security notes

- A self-hosted runner executes code from this repository on your
  hardware. Only run it against a repository you control, and do
  not add the `gpu` label to a runner shared with untrusted repos.
- The runner service account should be unprivileged. Do not run
  the runner as root.
- The RWT workflow never needs a secret beyond `GITHUB_TOKEN`:
  there are no API keys, no cloud credentials, and no model
  registry logins in any job. If a future step needs one, it
  belongs in GitHub Actions secrets, never in a file.
- Network: the runner needs outbound HTTPS to github.com (and, for
  the neural milestone, to wherever the base model comes from).
  It needs no inbound ports.
- The reference trainer runs on CPU inside the GPU job today
  (proving the slot works). The neural trainer that plugs into
  this slot later will need CUDA on the box; that is a hardware
  setup step (drivers + container runtime), not a repo change.

## If the GPU job stays queued

That means no Idle runner carries the `gpu` label. Check, in
order: the runner process is running on the box, the labels
include `gpu`, the runner is Idle (not Offline), and the workflow
was dispatched manually with `gpu` set to true. A queued GPU job
never blocks publishing: `publish` needs `evaluate`, not
`train-gpu`.
