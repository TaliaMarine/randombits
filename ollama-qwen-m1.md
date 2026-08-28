# Ollama + Qwen3.8-27B as a headless service on a MacBook Pro M1 Pro (32 GB), served to your LAN

The M1 Mac is treated as a **server**: no desktop app, no menu bar icon, no GUI toggles. One
launchd-managed `ollama serve` process, configured entirely from a plist, surviving reboots
unattended.

End-to-end:

1. Install the Ollama CLI only — and remove the desktop app if it's already there (§1)
2. Configure the service and its `OLLAMA_*` environment, persistently (§2)
3. Pull Qwen3.8-27B at a quantization that fits 32 GB, and bake settings into a model variant (§3)
4. Expose it to the LAN and keep the machine reachable across reboots (§4)
5. Point a remote laptop's coding agent at it (§5)
6. Build a small demo app — Go backend + Vue 3 + TypeScript frontend (§6)

Throughout, `SERVER_IP` means the LAN IP of the M1 Mac (e.g. `192.168.1.42`), and `$USER` is the
account that owns the service.

**Two rules this document follows.** Every setting is persistent — nothing that a reboot wipes is
presented as a solution, and where a persistent mechanism doesn't exist, that's stated. And every
setting is *verified*: `OLLAMA_*` variables fail silently, so each one comes with a way to prove
the running server actually picked it up.

---

## 0. The model

**Qwen3.8-27B** (<https://huggingface.co/Qwen/Qwen3.8-27B>), Apache-2.0. Relevant properties:

| Property | Value |
|---|---|
| Parameters | 27B language model (~28B incl. vision encoder) |
| Architecture | **Dense**, hybrid layer stack: 64 layers as `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))` |
| Modality | **Natively vision-language** (image-text-to-text) |
| Context | 262,144 native, extensible to ~1M via YaRN RoPE scaling |
| Reasoning | Thinking mode **on by default**, with a `reasoning_effort` control (`low` / `medium` / `xhigh`) and `preserve_thinking` on by default |
| Extras | Multi-token prediction (MTP) |
| Upstream weights | BF16; also an official `Qwen/Qwen3.8-27B-FP8` |

Two architectural details matter for a 32 GB laptop:

- **Only 1 layer in 4 is full attention.** The other three use Gated DeltaNet (linear attention, no
  growing KV cache). So the KV cache costs roughly a quarter of what a conventional dense 27B would
  need at the same context length. Long contexts are unusually cheap here — this is the single
  biggest reason the model is viable on 32 GB.
- **It is dense, not MoE.** Every one of the 27B params is read from memory for every token. On a
  memory-bandwidth-bound Mac that sets a hard speed ceiling (see §0.2). There is no MoE shortcut
  like Qwen3-30B-A3B had.

### 0.1 Which quantization

At 27B dense, the quant choice *is* the fit-or-not decision. Rough sizes (weights only, excluding
KV cache and the vision encoder):

| Quant | Size | Verdict on 32 GB |
|---|---|---|
| `Q3_K_M` | ~13 GB | Safety valve. Noticeably degraded; use only if Q4 won't stay on GPU. |
| **`Q4_K_M`** | **~16–17 GB** | **Recommended.** Leaves ~5 GB for KV cache + OS. The sweet spot. |
| `Q5_K_M` | ~19–20 GB | Tight. Works if you close everything else and raise the GPU limit (§4.5). |
| `Q6_K` | ~22–23 GB | Not advisable — leaves almost nothing for context. |
| `Q8_0` | ~29 GB | Won't fit. |
| BF16 | ~54 GB+ | Won't fit. |

Go with **`Q4_K_M`**.

A headless server has a real advantage here: with no Finder, no browser, no Xcode and no desktop
session holding memory, you get 2–4 GB back compared to the same machine used interactively. That
margin is the difference between Q4_K_M with a 32k context sitting comfortably on the GPU and it
spilling to CPU.

### 0.2 Speed expectations — read this before you're disappointed

Token generation for a dense model is bounded by `memory_bandwidth ÷ model_size`. The M1 Pro has
**~200 GB/s**, and Q4_K_M weights are ~16 GB, so:

- Theoretical ceiling: ~12 tok/s
- **Realistic: ~7–10 tok/s**

That is *usable for chat and single-file edits, but slow for agentic coding loops* — roughly reading
pace. Thinking mode makes it worse, since reasoning tokens cost wall-clock time before you see any
answer. Budget accordingly:

- Set `reasoning_effort` to `low`, or disable thinking, for anything interactive (§3.3).
- MTP-enabled GGUF builds advertise a speedup from multi-token prediction; worth trying if your
  Ollama build supports it.
- If it's too slow, the honest fix is a smaller model, not more tuning. A `qwen3:14b`-class model at
  ~9 GB runs roughly 2× faster on the same chip.

Sanity-check the chip with `sysctl -n machdep.cpu.brand_string` — it should report `Apple M1 Pro`.
(An M1 Max, at ~400 GB/s, would roughly double these figures.)

### 0.3 A faster alternative on Apple Silicon: MLX

MLX is Apple's own array framework and is frequently faster than llama.cpp on M-series hardware,
with better vision support. `mlx-community/Qwen3.8-27B-4bit` (~15–16 GB) exists and keeps the vision
capability.

It can be run headlessly — `mlx_lm.server` exposes an OpenAI-compatible API:

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen3.8-27B-4bit --host 0.0.0.0 --port 11434
```

The catch: it has no model registry, no `keep_alive` semantics, no `/api/tags`, and no equivalent of
the `OLLAMA_*` configuration surface — you manage the process and its flags yourself. If you want
the service story in §2 and §4, stay on Ollama. If you mainly care about raw speed, wrap the command
above in the same LaunchAgent pattern from §2.4 (swapping `ProgramArguments`) and skip to §5 —
everything from there on speaks the OpenAI-compatible API and works unchanged.

The third option, if Ollama can't load the architecture at all, is llama.cpp's own server:

```bash
brew install llama.cpp
llama-server --host 0.0.0.0 --port 11434 -m /path/to/model.gguf -c 32768 -ngl 999 -fa
```

llama.cpp gets support for new architectures first, and its flags are explicit rather than
environment-driven — which sidesteps §2 entirely at the cost of Ollama's model management.

---

## 1. Install the CLI only — no desktop app

### 1.1 Remove the desktop app if it's present

This matters before anything else: the desktop app and a service you manage yourself both want port
`11434`, and only one can have it. Worse, the app registers itself as a login item, so it silently
comes back after a reboot and wins the race — a very common cause of "my settings worked yesterday
and don't today".

```bash
# Is anything already holding the port?
lsof -nP -iTCP:11434 -sTCP:LISTEN

# Quit and remove the app
osascript -e 'quit app "Ollama"' 2>/dev/null
brew uninstall --cask ollama-app 2>/dev/null
rm -rf /Applications/Ollama.app
```

Then remove its autostart. Modern builds register via `SMAppService`, which does **not** show up as
a plist you can delete — it appears in **System Settings → General → Login Items & Extensions →
Open at Login**. Remove `Ollama` there. Older builds instead leave a plist behind:

```bash
ls ~/Library/LaunchAgents/ | grep -i ollama    # e.g. com.electron.ollama.plist
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.electron.ollama.plist 2>/dev/null
rm -f ~/Library/LaunchAgents/com.electron.ollama.plist
```

Your models in `~/.ollama/models` are untouched by all of this — the app and the CLI share that
directory.

### 1.2 Install the CLI

```bash
brew install ollama          # the formula — CLI + server binary, no GUI
```

Note the distinction: `brew install ollama` is the headless formula; `brew install --cask
ollama-app` is the desktop app. You want the former. Confirm you got a binary and no bundle:

```bash
which -a ollama              # /opt/homebrew/bin/ollama on Apple Silicon
ollama --version
```

Record that path — launchd does not use your shell's `PATH`, so §2.4 needs the absolute one.

### 1.3 Do not use `brew services`

`brew services start ollama` works, and it is the wrong tool here. Homebrew **regenerates** the
LaunchAgent plist from the formula definition every time you run `brew services start` or
`brew services restart`. Any `EnvironmentVariables` block you hand-edit into it is discarded on the
next restart, without warning.

If you have followed advice to edit `$(brew --prefix)/opt/ollama/homebrew.mxcl.ollama.plist`, that
is very likely why your `OLLAMA_*` settings won't stick — on formulae that declare a `service` block
(ollama does) that file often doesn't even exist, so you may have been editing a path that has no
effect at all.

So: stop it if it's running, and own the plist yourself (§2.4).

```bash
brew services stop ollama 2>/dev/null
brew services list | grep -i ollama    # should show "none" or nothing
```

### 1.4 Upgrade before pulling

Qwen3.8 uses a hybrid Gated DeltaNet stack — comparatively new in ggml terms. Support for novel
architectures lands in llama.cpp first and reaches Ollama on a lag. An older Ollama fails with an
`unknown model architecture` error.

```bash
brew upgrade ollama
ollama --version
```

Also check that your version even *knows* the variables you plan to set — the list has grown over
time, and Ollama silently ignores names it doesn't recognise:

```bash
ollama serve --help          # lists every OLLAMA_* variable this build understands
```

If `OLLAMA_CONTEXT_LENGTH` or `OLLAMA_KV_CACHE_TYPE` isn't in that output, your build is too old for
§2 and no amount of plist editing will help.

---

## 2. Configure the service and its environment — persistently

This is the section to read slowly. Ollama is configured almost entirely through environment
variables, and every one of them fails *silently*: a typo, a variable set in the wrong place, or a
server that was already running when you set it all produce the same symptom — the default value,
with no error anywhere.

### 2.1 Why your `OLLAMA_*` settings aren't taking effect

Four things go wrong, usually more than one at once.

**1. You set them for the wrong process.** `ollama` is two programs in one binary. `ollama run`,
`ollama pull`, `ollama ps` are *clients* that speak HTTP to a server. `ollama serve` is the
*server*. Nearly every `OLLAMA_*` variable — `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_FLASH_ATTENTION`,
`OLLAMA_KV_CACHE_TYPE`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_MODELS`, `OLLAMA_NUM_PARALLEL` — is read **only
by the server, only at startup**. Exporting them in your shell and then running `ollama run` sets
them on the *client*, where they do nothing at all.

`OLLAMA_HOST` is the confusing exception: the server reads it as *the address to bind*, and the
client reads it as *the address to connect to*. Same name, opposite meanings, depending on which
process sees it.

**2. You used a mechanism that doesn't reach a launchd job.** These are the options and what each
actually covers:

| Where you set it | Reaches the launchd-managed server? | Survives reboot? |
|---|---|---|
| `export` in `~/.zshrc` / `~/.zprofile` | **No** — launchd jobs don't source your shell | n/a |
| `launchctl setenv FOO bar` | Only if the job is (re)started afterwards | **No** — wiped on reboot |
| `EnvironmentVariables` in your own plist | **Yes** | **Yes** |
| `brew services` generated plist | Yes, until Homebrew regenerates it (§1.3) | Not reliably |
| Inline `FOO=bar ollama serve` in a terminal | Yes, for that process only | No — dies with the shell |

Only the third row satisfies both columns, which is why §2.4 is the whole answer. `launchctl setenv`
in particular is the trap: it appears to work, so you conclude the variable is fine, and then a
reboot silently reverts you to defaults.

**3. You didn't restart the server.** Ollama snapshots its configuration when `ollama serve` starts.
Changing the environment afterwards has no effect on the running process, ever. Every change in this
section requires a `launchctl kickstart -k` (§2.5).

**4. A client overrode you.** `OLLAMA_CONTEXT_LENGTH` sets a *default*. If a coding agent sends
`"options": {"num_ctx": 4096}` — and several do, without telling you — the request wins. See §2.3.

### 2.2 The variables that matter

Defaults are for recent Ollama builds; confirm yours with `ollama serve --help`.

| Variable | Set to | Default | Why |
|---|---|---|---|
| `OLLAMA_HOST` | `0.0.0.0:11434` | `127.0.0.1:11434` | Bind on all interfaces so the LAN can reach it (§4). |
| `OLLAMA_CONTEXT_LENGTH` | `32768` | `4096` (older: `2048`) | The default is tiny. Coding agents send huge prompts and get **silently truncated**, which reads as the model "forgetting" your files. |
| `OLLAMA_NUM_PARALLEL` | `1` | `0` (auto) | The one people miss. On auto, Ollama may run up to 4 concurrent slots and size the KV cache accordingly — so a 32k request can end up with 8k per slot, or allocate 4× the memory you budgeted. One user, one slot, no surprises. |
| `OLLAMA_FLASH_ATTENTION` | `1` | off / auto | Faster attention, less KV memory. Also a **prerequisite** for the next row. |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | `f16` | Quantizes the KV cache, roughly halving context memory for negligible quality loss. Valid values: `f16`, `q8_0`, `q4_0`. **Silently ignored unless flash attention is on.** |
| `OLLAMA_KEEP_ALIVE` | `24h` | `5m` | Avoids paying a 16 GB reload on every request after an idle gap. On a dedicated server there is no reason to unload; `-1` means never. |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | `0` (auto) | Stops a second model loading alongside the 27B and thrashing swap. |
| `OLLAMA_MODELS` | `/Users/$USER/.ollama/models` | `$HOME/.ollama/models` | Set it *explicitly*. Under launchd, `$HOME` is not always what you expect — a root LaunchDaemon resolves it to `/var/root`, and then your 16 GB pull lands somewhere you didn't intend, or fails on permissions. |
| `OLLAMA_ORIGINS` | `*` | localhost only | Only needed if browser JavaScript calls Ollama directly. The Go backend in §6 proxies server-side and does not need this. Leave it unset if you don't need it — it is a real widening of exposure. |
| `OLLAMA_LOAD_TIMEOUT` | `10m` | `5m` | A 16 GB first load from a cold page cache can exceed the default and be aborted mid-load. |
| `OLLAMA_DEBUG` | `1` *(temporarily)* | off | Turn on while diagnosing, then take it back out — it is extremely verbose. |

**On the 262k context:** do not set it. Native 262k is a property of the weights, not a budget you
have. Even with only a quarter of layers keeping a KV cache, a six-figure context needs far more
than the ~5 GB spare after a 16 GB model. Prefill time also grows with context, so on a 200 GB/s M1
Pro a huge window makes the first token painfully slow even when it fits. `32768` is the practical
ceiling here, not merely the memory-safe one.

### 2.3 Precedence — env var vs Modelfile vs request

For any given generation, `num_ctx` and sampling parameters are resolved in this order, later
winning:

```
OLLAMA_CONTEXT_LENGTH (server default)
  → PARAMETER lines in the model's Modelfile (§3.2)
    → "options" in the individual API request  ← highest priority
```

The practical consequences:

- If a client sends `num_ctx`, your server-wide setting is irrelevant *for that client*. Continue,
  Cline and some OpenAI-compatible wrappers do this by default.
- Baking `PARAMETER num_ctx 32768` into a named model variant (§3.2) is more robust than the
  environment variable, because it survives clients that don't let you set options — but it still
  loses to a client that sets them explicitly.
- Belt and braces: set the env var, bake the Modelfile parameter, *and* configure the client (§5).
  Then check what the running model actually got (§2.6).

### 2.4 The LaunchAgent — the one place configuration lives

Create `~/Library/LaunchAgents/local.ollama.serve.plist`. Substitute your real username for
`YOURUSER` and your real binary path from §1.2.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>local.ollama.serve</string>

  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/ollama</string>
    <string>serve</string>
  </array>

  <key>EnvironmentVariables</key>
  <dict>
    <key>OLLAMA_HOST</key>              <string>0.0.0.0:11434</string>
    <key>OLLAMA_CONTEXT_LENGTH</key>    <string>32768</string>
    <key>OLLAMA_NUM_PARALLEL</key>      <string>1</string>
    <key>OLLAMA_FLASH_ATTENTION</key>   <string>1</string>
    <key>OLLAMA_KV_CACHE_TYPE</key>     <string>q8_0</string>
    <key>OLLAMA_KEEP_ALIVE</key>        <string>24h</string>
    <key>OLLAMA_MAX_LOADED_MODELS</key> <string>1</string>
    <key>OLLAMA_LOAD_TIMEOUT</key>      <string>10m</string>
    <key>OLLAMA_MODELS</key>            <string>/Users/YOURUSER/.ollama/models</string>
    <key>HOME</key>                     <string>/Users/YOURUSER</string>
    <key>PATH</key>                     <string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>

  <key>RunAtLoad</key>  <true/>
  <key>KeepAlive</key>  <true/>

  <key>StandardOutPath</key>   <string>/Users/YOURUSER/Library/Logs/ollama.log</string>
  <key>StandardErrorPath</key> <string>/Users/YOURUSER/Library/Logs/ollama.log</string>
</dict>
</plist>
```

Points that trip people up:

- **Every value must be a `<string>`**, including numbers and booleans. `<integer>1</integer>` for
  `OLLAMA_FLASH_ATTENTION` will not be delivered as the environment variable you expect.
- **Absolute paths only.** launchd's `PATH` does not include `/opt/homebrew/bin`; a bare `ollama` in
  `ProgramArguments` yields a job that dies instantly with status 78.
- **No `~` expansion** anywhere in a plist. Write `/Users/YOURUSER/...` in full.
- **`KeepAlive`** restarts the server if it crashes or is killed — which is what makes it a service
  rather than something you have to remember.
- **`OLLAMA_ORIGINS` is deliberately absent.** Add `<key>OLLAMA_ORIGINS</key><string>*</string>`
  only if a browser talks to Ollama directly.
- The log path must exist as a directory: `~/Library/Logs` always does.

Validate the XML before loading — a malformed plist is rejected wholesale and is the single most
common reason "nothing happens":

```bash
plutil -lint ~/Library/LaunchAgents/local.ollama.serve.plist    # must print "OK"
```

### 2.5 Load, restart, inspect

Modern `launchctl` subcommands (`load`/`unload` are deprecated and give worse errors):

```bash
# Load it and start it now
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.ollama.serve.plist

# Apply any config change — this is mandatory after editing the plist
launchctl kickstart -k gui/$(id -u)/local.ollama.serve

# State, last exit code, and the environment launchd will hand it
launchctl print gui/$(id -u)/local.ollama.serve

# Stop and unload entirely
launchctl bootout gui/$(id -u)/local.ollama.serve
```

`launchctl print` is the authoritative view of what launchd *intends* to pass. If a variable is
missing from its `environment` block, the plist is at fault, not Ollama.

Note that `launchctl bootstrap` fails with `Bootstrap failed: 5: Input/output error` if the job is
already loaded — `bootout` first, then `bootstrap` again.

### 2.6 Verify the server actually received them

Ollama logs its **entire resolved configuration** on startup. This is the single most useful
debugging fact in this document:

```bash
grep -a 'server config' ~/Library/Logs/ollama.log | tail -1 | tr ' ' '\n' | grep OLLAMA_
```

You get one line per variable with the value the running server is genuinely using:

```
OLLAMA_CONTEXT_LENGTH:32768
OLLAMA_FLASH_ATTENTION:true
OLLAMA_HOST:http://0.0.0.0:11434
OLLAMA_KEEP_ALIVE:24h0m0s
OLLAMA_KV_CACHE_TYPE:q8_0
OLLAMA_MAX_LOADED_MODELS:1
OLLAMA_MODELS:/Users/YOURUSER/.ollama/models
OLLAMA_NUM_PARALLEL:1
```

If a value here is the default, the variable never arrived — stop editing Ollama config and go back
to §2.4/§2.5. If the value is right here but behaviour still looks wrong, the problem is downstream:
a client override (§2.3) or a per-model parameter (§3.2).

Then check the *effective* context of a loaded model, which is what actually matters:

```bash
curl -s http://127.0.0.1:11434/api/ps | jq '.models[] | {name, context_length, size_vram}'

# And what the inference runner allocated, per slot:
grep -a 'n_ctx' ~/Library/Logs/ollama.log | tail -5
```

A `context_length` of 4096 when the server config says 32768 is the §2.3 override, near-certainly.

Live log while testing:

```bash
tail -f ~/Library/Logs/ollama.log
```

(Note: `~/.ollama/logs/server.log` is written by the *desktop app*. With the service you own the log
path, and it's whatever `StandardErrorPath` says.)

### 2.7 Starting at boot with nobody logged in

A LaunchAgent starts when the user session starts — i.e. at **login**, not at boot. For an unattended
server that is a gap: after a power cut the machine reboots to the login window and Ollama never
starts. Two ways to close it.

**Option A — LaunchAgent plus automatic login (recommended).** Enable **System Settings → Users &
Groups → Automatically log in as** for the service account. The session comes up on its own after
any reboot, and the agent with it. This keeps Ollama in a normal GUI session, which is the
configuration Metal GPU acceleration is best tested in.

Caveat: **FileVault blocks automatic login.** With FileVault on, the disk stays locked until someone
types a password at the login screen, so nothing starts unattended. For a machine that lives on your
LAN, either accept that reboots need one manual unlock, or turn FileVault off knowing what you're
giving up (full-disk encryption on a portable machine). There is no third option — this is a
deliberate macOS constraint, not a setting to work around.

**Option B — LaunchDaemon.** Put the same plist in `/Library/LaunchDaemons/`, owned by root, and it
starts at boot with no session at all:

```bash
sudo cp ~/Library/LaunchAgents/local.ollama.serve.plist /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/local.ollama.serve.plist
sudo chmod 644 /Library/LaunchDaemons/local.ollama.serve.plist
# add <key>UserName</key><string>YOURUSER</string> so it runs as you, not root,
# and keeps using /Users/YOURUSER/.ollama/models
sudo launchctl bootstrap system /Library/LaunchDaemons/local.ollama.serve.plist
```

Ownership and permissions are enforced: a daemon plist that is group- or world-writable is refused
without a useful message.

The catch worth knowing: processes in the system context have no window-server session, and GPU
access from there is not guaranteed on every macOS version. Verify explicitly after switching —
`ollama ps` must report `100% GPU` (§3.1). If it reports any CPU percentage that it didn't before,
revert to Option A; a 27B model on CPU is unusably slow.

Run one or the other, never both — two servers, one port.

### 2.8 If it still won't start

```bash
launchctl print gui/$(id -u)/local.ollama.serve | grep -E 'state|last exit'
```

| Last exit status | Meaning |
|---|---|
| `78` | Bad `ProgramArguments` path — not an absolute path to a real binary. |
| `1` | Ollama started and exited. Read the log; usually the port is taken (§1.1) or `OLLAMA_MODELS` is unwritable. |
| `126` / `127` | Not executable / not found. Check `ls -l` on the binary. |
| No such job | The plist failed to parse or was never bootstrapped. `plutil -lint`, then bootstrap again. |

Two more that produce baffling behaviour:

- **Something else is already on 11434.** `lsof -nP -iTCP:11434 -sTCP:LISTEN` and check the process
  name. If it's `Ollama` (capital O) you did not finish §1.1.
- **Variable names are case-sensitive and exact.** `OLLAMA_CONTEXT_SIZE`, `OLLAMA_NUM_CTX` and
  `OLLAMA_KEEPALIVE` are not real variables; they are accepted into the environment and ignored
  forever. Cross-check every name against `ollama serve --help`, and confirm against §2.6.

---

## 3. Pull the model

The service must be running before you pull — `ollama pull` is a client and talks to the server, which
is what actually writes to `OLLAMA_MODELS`.

```bash
curl -s http://127.0.0.1:11434/api/version
```

### 3.1 Pull and smoke-test

I was unable to verify the exact Ollama library tag from this machine (corporate proxy blocks
`ollama.com`), so **check <https://ollama.com/library/qwen3.8> for the current tag list first**. If
an official tag exists it will look like:

```bash
ollama pull qwen3.8:27b
# or an explicit quant, if published:
ollama pull qwen3.8:27b-q4_K_M
```

**Fallback that works regardless** — Ollama can pull GGUF straight from HuggingFace:

```bash
ollama pull hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_K_M
```

`unsloth/Qwen3.8-27B-GGUF` is the highest-traffic community conversion (~7.8M downloads) and a safe
default. Alternatives if it gives you trouble: `empero-ai/Qwen3.8-27B-Ridge-GGUF`,
`incoai/Qwen3.8-27B-DFlash2-GGUF`.

> **Two cautions on community GGUFs.** (1) Many repos in the search results are "uncensored" /
> "abliterated" / "heretic" finetunes — those are behaviour-modified, not just requantized. Unless
> you specifically want that, stick to a plain conversion of the official weights. (2) Not every GGUF
> conversion keeps the **vision** capability; that requires a separate `mmproj` projector file. Repos
> tagged `image-text-to-text` retain it, text-only conversions don't. If you don't need image input,
> don't pay the complexity.

Smoke test over the API rather than interactively — on a headless box that's the thing you actually
care about:

```bash
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_K_M",
  "prompt": "Write a Go function that reverses a UTF-8 string correctly.",
  "stream": false
}' | jq -r '.response'
```

Then confirm it's on the GPU:

```bash
ollama ps
```

`PROCESSOR` must read `100% GPU`. Any CPU percentage means the model is spilling — see §4.5, or drop
to `Q3_K_M`.

### 3.2 Bake settings into a named model

Do this even though §2 set the environment. A named variant carries its context and sampling with it,
so clients that don't expose `num_ctx` still get the right window (§2.3), and the short name is far
nicer than an `hf.co/...` tag:

```bash
cat > /tmp/Modelfile <<'EOF'
FROM hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_K_M
PARAMETER num_ctx 32768
PARAMETER temperature 0.7
PARAMETER top_p 0.80
EOF

ollama create qwen38-code -f /tmp/Modelfile
```

Verify the parameters stuck — `show` is the non-interactive equivalent of `/show parameters`:

```bash
ollama show qwen38-code --parameters
ollama show qwen38-code             # architecture, context limit, license
```

The rest of this document uses `qwen38-code`. Note that `ollama create` does not copy the 16 GB — the
variant references the same blobs.

### 3.3 Thinking mode and sampling

Qwen3.8 thinks by default and preserves its reasoning trace. Two consequences: output is slower, and
`<think>` blocks may leak into your application's text unless the client strips them (§6.1 does).

Sampling settings recommended by the model card — they differ per mode, and using thinking-mode
settings in non-thinking mode (or vice versa) measurably degrades output:

| Mode | temperature | top_p | presence_penalty |
|---|---|---|---|
| Thinking | 1.0 | 0.95 | — |
| Non-thinking | 0.7 | 0.80 | 1.5 |

The Modelfile above uses non-thinking values, which is the right default for interactive work on this
hardware. To control effort, prefer `reasoning_effort: low` if your client and the GGUF's chat
template expose it; otherwise fall back to appending `/no_think` to the prompt — the convention Qwen
models have honoured since Qwen3. Test which one your conversion actually respects rather than
assuming; template support varies between GGUF repos.

```bash
curl -s http://127.0.0.1:11434/api/generate -d '{
  "model":"qwen38-code","prompt":"2+2? /no_think","stream":false
}' | jq -r '.response'    # no <think> block means the template honoured it
```

---

## 4. Expose it to the LAN, permanently

§2 already bound the server to `0.0.0.0:11434` via the plist — there is no GUI toggle to hunt for and
nothing to redo after a reboot. What remains is the surrounding host configuration.

### 4.1 Confirm the bind

```bash
lsof -nP -iTCP:11434 -sTCP:LISTEN
# want *:11434 — 127.0.0.1:11434 means OLLAMA_HOST didn't arrive (§2.6)
```

### 4.2 Firewall

macOS's application firewall persists its rules, so this is one-time:

```bash
# Inspect current state
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# Allow the server binary explicitly (persistent, no GUI needed)
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /opt/homebrew/bin/ollama
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /opt/homebrew/bin/ollama
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --listapps | grep -i ollama
```

The GUI equivalent is **System Settings → Network → Firewall → Options**, setting `ollama` to *Allow
incoming connections*. Either is durable. Note that upgrading Ollama replaces the binary, and the
rule is per-binary-path — if remote access breaks right after a `brew upgrade`, re-run the two
commands above.

### 4.3 A stable address, and a machine that stays awake

A sleeping laptop serves no tokens, and `caffeinate` is exactly the kind of non-persistent hack this
document avoids — it dies with its terminal. `pmset` writes to persistent power management settings
instead:

```bash
sudo pmset -c sleep 0 displaysleep 0 disksleep 0   # on AC: never sleep
sudo pmset -c autorestart 1                        # reboot automatically after a power failure
sudo pmset -c womp 1                               # wake on network access
sudo pmset -a disablesleep 1                       # keep running with the lid closed

pmset -g custom                                    # verify; survives reboots
```

`disablesleep 1` is what makes clamshell operation work without an external display attached. Undo it
with `sudo pmset -a disablesleep 0`.

Then find the address and pin it:

```bash
ipconfig getifaddr en0      # Wi-Fi
ipconfig getifaddr en1      # Ethernet / dongle
```

Set a **DHCP reservation** for the Mac in your router so `SERVER_IP` never moves. Wired Ethernet is
worth it if you can — a 27B model streaming over congested Wi-Fi adds latency you don't need, and
Wi-Fi power management is a common cause of a server that becomes unreachable after hours of idle.

### 4.4 Verify from the remote laptop

```bash
curl http://SERVER_IP:11434/api/version
curl http://SERVER_IP:11434/api/tags | jq '.models[].name'

# Native Ollama chat API
curl http://SERVER_IP:11434/api/chat -d '{
  "model": "qwen38-code",
  "messages": [{"role":"user","content":"Say hello in exactly three words."}],
  "stream": false
}'

# OpenAI-compatible endpoint — what most coding agents use
curl http://SERVER_IP:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer ollama' \
  -d '{
    "model": "qwen38-code",
    "messages": [{"role":"user","content":"Say hello in exactly three words."}]
  }'
```

If `/api/version` works but `/v1/...` doesn't, your Ollama is too old — upgrade (§1.4).

Finally, prove the whole thing is actually a service: reboot the Mac, log in to nothing, and run the
first `curl` above from the laptop. If that works, you're done. If it doesn't, you're in §2.7.

### 4.5 If you see CPU spill

macOS reserves a fraction of unified memory for the GPU — on 32 GB that's roughly 21–24 GB by
default. A 16–17 GB model plus a 32k KV cache fits, but the margin is thin.

Cheaper things to try first, in order: confirm `OLLAMA_NUM_PARALLEL=1` actually took (§2.6 — this
alone can quadruple KV allocation), confirm `OLLAMA_KV_CACHE_TYPE=q8_0` took *and* that flash
attention is on, drop `OLLAMA_CONTEXT_LENGTH` to `16384`, then drop to `Q3_K_M`.

Only then raise the GPU's wired-memory limit:

```bash
sudo sysctl iogpu.wired_limit_mb=28672   # 28 GB of 32 GB. Leaves 4 GB for macOS.
sudo sysctl iogpu.wired_limit_mb=0       # restore the OS default
```

> **Caution:** starving macOS of RAM causes beachballs, heavy swap, or a kernel panic. Raise this only
> if `ollama ps` shows spill, do it in steps, and do not exceed 28672 on a 32 GB machine. `0` always
> restores the default.

`sysctl` is not persistent, and `/etc/sysctl.conf` is **not** reliably read by current macOS versions
— advice to put it there is stale. Use a LaunchDaemon, `/Library/LaunchDaemons/local.iogpu.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>local.iogpu</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/sbin/sysctl</string>
    <string>iogpu.wired_limit_mb=28672</string>
  </array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
```

```bash
sudo chown root:wheel /Library/LaunchDaemons/local.iogpu.plist
sudo chmod 644 /Library/LaunchDaemons/local.iogpu.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/local.iogpu.plist
sysctl iogpu.wired_limit_mb    # verify: 28672
```

It runs once at boot and exits — no `KeepAlive`, or launchd will respawn it in a loop.

### 4.6 Security

`0.0.0.0:11434` means **no authentication**. Anyone on the network can use your GPU and read anything
you send it. On a home LAN behind NAT that's usually acceptable. What to avoid:

- **Never** port-forward 11434 from your router to the internet.
- To reach it from outside, use **Tailscale**: install on both machines and bind to the tailnet
  address instead of `0.0.0.0` by setting `OLLAMA_HOST` to `100.x.y.z:11434` in the plist (§2.4).
  Encrypted, authenticated, stable hostname — strictly better than a LAN IP. Note that if you bind
  *only* to the tailnet address, the local CLI needs `OLLAMA_HOST=100.x.y.z:11434` exported in your
  shell too, since its default of `127.0.0.1:11434` will no longer be listening.
  Install Tailscale headlessly via `brew install tailscale` and `sudo brew services start tailscale`
  — the cask is a GUI app.
- If you need auth on the LAN, put Caddy or nginx in front with TLS and basic auth, and set
  `OLLAMA_HOST` back to `127.0.0.1:11434` so only the proxy can reach it.

---

## 5. Point a remote laptop's coding agent at it

Nothing here is installed on the server — the Mac is finished after §4.

### 5.1 opencode (recommended)

Install on the **remote laptop**:

```bash
curl -fsSL https://opencode.ai/install | bash
# or: npm i -g opencode-ai
# or: brew install sst/tap/opencode
```

Configure `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (M1 Mac)",
      "options": {
        "baseURL": "http://SERVER_IP:11434/v1"
      },
      "models": {
        "qwen38-code": {
          "name": "Qwen3.8 27B",
          "tools": true,
          "options": {
            "num_ctx": 32768,
            "temperature": 0.7,
            "top_p": 0.8
          }
        }
      }
    }
  },
  "model": "ollama/qwen38-code"
}
```

Notes:

- Replace `SERVER_IP`. The `/v1` suffix is required.
- `"npm": "@ai-sdk/openai-compatible"` selects the OpenAI-compatible adapter; opencode fetches that
  package on first run.
- Ollama ignores the API key, so no credentials are needed.
- Models must be listed explicitly — opencode does not auto-discover Ollama's catalogue.
- `"tools": true` is essential; without it opencode won't let the model edit files.
- The `num_ctx` here is a client-side override and therefore *wins* over the server (§2.3). Keep it
  in step with `OLLAMA_CONTEXT_LENGTH`, or the machine you carefully configured gets ignored.

Run it in a project:

```bash
cd ~/some/project
opencode
```

`/models` to confirm the provider appears; `/model ollama/qwen38-code` to switch. If the provider is
missing, rerun with `OPENCODE_LOG_LEVEL=debug` and look for config parse errors.

Commit a project-level `opencode.json` with just the `"model"` key so teammates on faster hardware can
choose differently.

### 5.2 Alternatives

**Continue.dev** (VS Code / JetBrains) — `~/.continue/config.yaml`:

```yaml
models:
  - name: Qwen3.8 27B
    provider: ollama
    model: qwen38-code
    apiBase: http://SERVER_IP:11434
    defaultCompletionOptions:
      contextLength: 32768
    roles: [chat, edit, apply]
```

No `/v1` for the native `ollama` provider — Continue appends the right paths itself. Set
`contextLength` explicitly: Continue sends a conservative default otherwise, and that is the
single most common source of "the server says 32k but the model behaves like it has 4k" (§2.3).
Don't put a 27B dense model in the `autocomplete` role; it's far too slow for keystroke latency. Pull
a 1.7B–8B model alongside it for that — and remember `OLLAMA_MAX_LOADED_MODELS=1` (§2.2) will make
the two evict each other, so raise it to `2` in the plist if you want both resident, memory
permitting.

**Zed** — `settings.json`:

```json
{
  "language_models": {
    "ollama": { "api_url": "http://SERVER_IP:11434" }
  },
  "agent": {
    "default_model": { "provider": "ollama", "model": "qwen38-code" }
  }
}
```

**aider**:

```bash
export OLLAMA_API_BASE=http://SERVER_IP:11434
aider --model ollama_chat/qwen38-code
```

Use `ollama_chat/`, not `ollama/` — the latter hits the completion endpoint and gives noticeably worse
results. aider also defaults to a small context; set `num_ctx` in `~/.aider.model.settings.yml` or
rely on the Modelfile parameter from §3.2.

**Cline / Roo Code** — API Provider `Ollama`, Base URL `http://SERVER_IP:11434`, then pick the model.

**Any OpenAI-compatible client** — Base URL `http://SERVER_IP:11434/v1`, API key any non-empty string.

### 5.3 Expectations

At ~7–10 tok/s on an M1 Pro, a 27B dense model is a capable assistant but a frustrating autonomous
agent — a 20-tool-call loop takes minutes of wall clock. What works well: single-file edits, writing
tests, explaining unfamiliar code, boilerplate, scoped refactors. What doesn't: long multi-file
agentic runs, subtle concurrency bugs, staying coherent over very long sessions. Keep tasks scoped,
turn thinking down for interactive work (§3.3), and review the diffs.

---




## 6. Demo app — Go backend + Vue 3 / TypeScript frontend

A minimal streaming chat app. The Go backend proxies to Ollama — keeping the model host server-side, so no CORS handling and no need to expose Ollama to browsers — and the Vue frontend renders tokens as they arrive. The backend also strips `<think>` blocks, which you'll want given thinking mode is on by default.

```
qwen-demo/
├── backend/
│   ├── go.mod
│   └── main.go
└── frontend/
    ├── vite.config.ts
    └── src/
        ├── main.ts
        ├── App.vue
        └── api.ts
```

### 6.1 Backend

```bash
mkdir -p qwen-demo/backend && cd qwen-demo/backend
go mod init qwen-demo/backend
```

`backend/main.go`:

```go
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Messages []chatMessage `json:"messages"`
}

type ollamaRequest struct {
	Model    string         `json:"model"`
	Messages []chatMessage  `json:"messages"`
	Stream   bool           `json:"stream"`
	Options  map[string]any `json:"options,omitempty"`
}

type ollamaChunk struct {
	Message chatMessage `json:"message"`
	Done    bool        `json:"done"`
	Error   string      `json:"error,omitempty"`
}

type server struct {
	ollamaURL string
	model     string
	client    *http.Client
}

// thinkFilter drops <think>…</think> spans from a token stream. Qwen3.8 has
// thinking enabled by default, and the tags arrive split across chunks, so
// this has to buffer across calls rather than filter each chunk in isolation.
type thinkFilter struct {
	inside bool
	buf    strings.Builder
}

func (f *thinkFilter) push(chunk string) string {
	f.buf.WriteString(chunk)
	var out strings.Builder
	for {
		s := f.buf.String()
		if f.inside {
			end := strings.Index(s, "</think>")
			if end < 0 {
				// Keep a short tail in case a closing tag straddles chunks.
				if len(s) > 16 {
					f.buf.Reset()
					f.buf.WriteString(s[len(s)-16:])
				}
				return out.String()
			}
			f.inside = false
			f.buf.Reset()
			f.buf.WriteString(s[end+len("</think>"):])
			continue
		}
		start := strings.Index(s, "<think>")
		if start < 0 {
			// Hold back a possible partial "<think" prefix.
			safe := len(s)
			if i := strings.LastIndexByte(s, '<'); i >= 0 && len(s)-i < len("<think>") {
				safe = i
			}
			out.WriteString(s[:safe])
			f.buf.Reset()
			f.buf.WriteString(s[safe:])
			return out.String()
		}
		out.WriteString(s[:start])
		f.inside = true
		f.buf.Reset()
		f.buf.WriteString(s[start+len("<think>"):])
	}
}

func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req chatRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	if len(req.Messages) == 0 {
		http.Error(w, "messages must not be empty", http.StatusBadRequest)
		return
	}

	// Non-thinking sampling settings per the Qwen3.8 model card.
	body, err := json.Marshal(ollamaRequest{
		Model:    s.model,
		Messages: req.Messages,
		Stream:   true,
		Options: map[string]any{
			"num_ctx":          32768,
			"temperature":      0.7,
			"top_p":            0.8,
			"presence_penalty": 1.5,
		},
	})
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	upstream, err := http.NewRequestWithContext(
		r.Context(), http.MethodPost, s.ollamaURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	upstream.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(upstream)
	if err != nil {
		log.Printf("ollama unreachable: %v", err)
		http.Error(w, "model backend unreachable", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("ollama returned %s", resp.Status)
		http.Error(w, "model backend error", http.StatusBadGateway)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	// Ollama streams newline-delimited JSON; re-emit as SSE for the browser.
	filter := &thinkFilter{}
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var chunk ollamaChunk
		if err := json.Unmarshal(line, &chunk); err != nil {
			continue
		}
		if chunk.Error != "" {
			fmt.Fprintf(w, "event: error\ndata: %s\n\n", mustJSON(chunk.Error))
			flusher.Flush()
			return
		}
		if text := filter.push(chunk.Message.Content); text != "" {
			fmt.Fprintf(w, "data: %s\n\n", mustJSON(text))
			flusher.Flush()
		}
		if chunk.Done {
			fmt.Fprint(w, "event: done\ndata: {}\n\n")
			flusher.Flush()
			return
		}
	}
	if err := scanner.Err(); err != nil {
		log.Printf("stream read error: %v", err)
	}
}

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return []byte(`""`)
	}
	return b
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	resp, err := s.client.Get(s.ollamaURL + "/api/version")
	if err != nil {
		http.Error(w, `{"ok":false}`, http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"ok":true,"model":%s}`, mustJSON(s.model))
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	s := &server{
		ollamaURL: env("OLLAMA_URL", "http://127.0.0.1:11434"),
		model:     env("OLLAMA_MODEL", "qwen38-code"),
		// No global timeout: a 27B dense model on an M1 can legitimately take
		// minutes. Cancellation comes from the request context on disconnect.
		client: &http.Client{},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api/chat", s.handleChat)
	mux.HandleFunc("/api/health", s.handleHealth)

	addr := env("ADDR", "127.0.0.1:8080")
	log.Printf("listening on %s, proxying to %s (model %s)", addr, s.ollamaURL, s.model)

	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Fatal(srv.ListenAndServe())
}
```

Run it — on the M1 Mac itself:

```bash
cd qwen-demo/backend && go run .
```

Or from another machine, pointing at the remote Ollama:

```bash
OLLAMA_URL=http://SERVER_IP:11434 OLLAMA_MODEL=qwen38-code go run .
```

Check:

```bash
curl http://127.0.0.1:8080/api/health
curl -N http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to five."}]}'
```

`-N` disables curl's buffering so you see tokens stream in.

### 6.2 Frontend

```bash
cd qwen-demo
npm create vite@latest frontend -- --template vue-ts
cd frontend && npm install
```

`frontend/vite.config.ts`:

```ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8080', changeOrigin: true },
    },
  },
})
```

`frontend/src/api.ts`:

```ts
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

/** POSTs the conversation and yields assistant tokens as they arrive. */
export async function* streamChat(
  messages: ChatMessage[],
  signal: AbortSignal,
): AsyncGenerator<string> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
    signal,
  })

  if (!res.ok || !res.body) {
    throw new Error(`request failed: ${res.status} ${await res.text()}`)
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value

    // SSE frames are separated by a blank line.
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)

      let event = 'message'
      const dataLines: string[] = []
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
      }
      if (!dataLines.length) continue

      const payload = dataLines.join('\n')
      if (event === 'done') return
      if (event === 'error') throw new Error(JSON.parse(payload))
      yield JSON.parse(payload) as string
    }
  }
}
```

`frontend/src/App.vue`:

```vue
<script setup lang="ts">
import { nextTick, ref, useTemplateRef } from 'vue'
import { streamChat, type ChatMessage } from './api'

const messages = ref<ChatMessage[]>([])
const draft = ref('')
const busy = ref(false)
const error = ref('')
const log = useTemplateRef<HTMLDivElement>('log')
let controller: AbortController | null = null

async function scrollToBottom() {
  await nextTick()
  log.value?.scrollTo({ top: log.value.scrollHeight })
}

async function send() {
  const text = draft.value.trim()
  if (!text || busy.value) return

  draft.value = ''
  error.value = ''
  busy.value = true
  messages.value.push({ role: 'user', content: text })

  const reply: ChatMessage = { role: 'assistant', content: '' }
  messages.value.push(reply)
  await scrollToBottom()

  controller = new AbortController()
  try {
    for await (const token of streamChat(
      messages.value.slice(0, -1),
      controller.signal,
    )) {
      reply.content += token
      await scrollToBottom()
    }
  } catch (e) {
    if ((e as Error).name !== 'AbortError') error.value = (e as Error).message
  } finally {
    busy.value = false
    controller = null
  }
}

function stop() {
  controller?.abort()
}
</script>

<template>
  <main>
    <h1>Qwen3.8 · local</h1>

    <div ref="log" class="log">
      <p v-if="!messages.length" class="hint">Ask the model something.</p>
      <article v-for="(m, i) in messages" :key="i" :class="m.role">
        <strong>{{ m.role === 'user' ? 'You' : 'Qwen' }}</strong>
        <pre>{{ m.content || '…' }}</pre>
      </article>
    </div>

    <p v-if="error" class="error">{{ error }}</p>

    <form @submit.prevent="send">
      <textarea
        v-model="draft"
        placeholder="Ask something… (Enter to send, Shift+Enter for newline)"
        @keydown.enter.exact.prevent="send"
      />
      <button v-if="busy" type="button" @click="stop">Stop</button>
      <button v-else type="submit" :disabled="!draft.trim()">Send</button>
    </form>
  </main>
</template>

<style scoped>
main {
  max-width: 46rem;
  margin: 0 auto;
  padding: 1.5rem;
  font: 15px/1.6 ui-sans-serif, system-ui, sans-serif;
}
h1 { font-size: 1.1rem; letter-spacing: 0.02em; text-transform: uppercase; opacity: 0.6; }
.log {
  height: 60vh;
  overflow-y: auto;
  border: 1px solid color-mix(in oklch, currentColor 15%, transparent);
  border-radius: 10px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.hint { opacity: 0.45; }
article { display: grid; gap: 0.25rem; }
article strong { font-size: 0.72rem; text-transform: uppercase; opacity: 0.5; }
article.user pre { background: color-mix(in oklch, currentColor 7%, transparent); }
pre {
  margin: 0;
  padding: 0.6rem 0.75rem;
  border-radius: 8px;
  white-space: pre-wrap;
  word-break: break-word;
  font: inherit;
}
.error { color: #c0392b; font-size: 0.85rem; }
form { display: flex; gap: 0.5rem; margin-top: 1rem; }
textarea {
  flex: 1; min-height: 3.4rem; resize: vertical; padding: 0.6rem; border-radius: 8px;
  border: 1px solid color-mix(in oklch, currentColor 20%, transparent); font: inherit;
}
button {
  padding: 0 1.1rem; border-radius: 8px; border: 0; cursor: pointer;
  background: currentColor; color: canvas; font: inherit;
}
button:disabled { opacity: 0.4; cursor: default; }
</style>
```

`frontend/src/main.ts`:

```ts
import { createApp } from 'vue'
import App from './App.vue'

createApp(App).mount('#app')
```

### 6.3 Run both

```bash
# Terminal 1
cd qwen-demo/backend && OLLAMA_URL=http://SERVER_IP:11434 go run .

# Terminal 2
cd qwen-demo/frontend && npm run dev
```

Open <http://localhost:5173>. The first token takes several seconds if the model isn't resident (16 GB read from disk); after that `OLLAMA_KEEP_ALIVE` keeps it warm. Expect the reply to render at reading pace, not instantly.

To serve the built frontend from Go, `npm run build`, then add to `main.go`:

```go
mux.Handle("/", http.FileServer(http.Dir("../frontend/dist")))
```

That serves `index.html` only at `/`; a Vue Router app in history mode needs a catch-all falling back to `index.html`.

---

## 7. Troubleshooting

### 7.1 Configuration and service

| Symptom | Cause / fix |
|---|---|
| An `OLLAMA_*` setting has no effect | Check it arrived: §2.6. If it's absent from the `server config` log line, the plist is at fault (§2.4) — not Ollama. If it's present but behaviour is wrong, a client is overriding it (§2.3). |
| Settings worked, then stopped after a reboot | You used `launchctl setenv`, an `export` in a shell, or an inline `FOO=bar ollama serve`. None of those persist. Move them into the plist (§2.4). |
| Settings worked, then stopped after `brew services restart` | Homebrew regenerated the plist and discarded your edits (§1.3). Stop `brew services` and use your own LaunchAgent. |
| Settings worked, then stopped after `brew upgrade` | Two possibilities: the firewall rule is per-binary-path (§4.2), or the desktop app got reinstalled and grabbed the port (§1.1). |
| Variable is in the plist but missing from `launchctl print` | Malformed plist, or a non-`<string>` value type. `plutil -lint`, then §2.4's list of gotchas. |
| Job won't start at all | `launchctl print gui/$(id -u)/local.ollama.serve` and grep `last exit` and look it up in §2.8. Exit 78 is almost always a non-absolute binary path. |
| Nothing starts after an unattended reboot | A LaunchAgent needs a login session. Enable automatic login, or use a LaunchDaemon — and mind the FileVault constraint (§2.7). |
| Two servers fighting / config seems to flip-flop | The desktop app is still installed and racing your service for port 11434. Finish §1.1, including the Login Items entry. |
| Models pulled but `ollama list` is empty | `OLLAMA_MODELS` differs between the client's view and the server's, or a root daemon wrote to `/var/root/.ollama`. Set it explicitly in the plist and check §2.6. |

### 7.2 Model and performance

| Symptom | Cause / fix |
|---|---|
| `unknown model architecture` on pull or run | Ollama too old for Qwen3.8's Gated DeltaNet stack. `brew upgrade ollama` (§1.4); if it persists, use `llama.cpp`'s `llama-server`, which gets architecture support first (§0.3). |
| `connection refused` from remote | `OLLAMA_HOST` didn't arrive. `lsof -nP -iTCP:11434 -sTCP:LISTEN` should show `*:11434`, not `127.0.0.1:11434`. Verify with §2.6, then `launchctl kickstart -k`. |
| Works locally, times out remotely | Firewall (§4.2), or the machine went to sleep (§4.3). |
| Unreachable after hours of idle | Sleep or Wi-Fi power management. `pmset -g custom` to confirm §4.3 took; prefer wired Ethernet. |
| Very slow (< 5 tok/s) | `ollama ps` showing CPU spill, or thinking mode burning tokens. See §4.5 and §3.3. Note ~7–10 tok/s is *expected* on M1 Pro — see §0.2. |
| Out of memory / kernel panic | Q4_K_M plus a large context exceeds the GPU budget. Check `OLLAMA_NUM_PARALLEL=1` first (§2.2), then lower `OLLAMA_CONTEXT_LENGTH`, then drop to `Q3_K_M`. Don't raise `iogpu.wired_limit_mb` past 28672. |
| Asked for 32k, got 8k | `OLLAMA_NUM_PARALLEL` on auto divided the window across slots. Set it to `1` (§2.2) and confirm via `/api/ps` (§2.6). |
| `OLLAMA_KV_CACHE_TYPE` seems ignored | It requires flash attention. Set `OLLAMA_FLASH_ATTENTION=1` too, and check both in §2.6. |
| Model reloads on every request | `OLLAMA_KEEP_ALIVE` too short or unset, or a second model evicting it. Set `24h` and `OLLAMA_MAX_LOADED_MODELS=1` (§2.2). |
| First load times out | Raise `OLLAMA_LOAD_TIMEOUT` — 16 GB from a cold page cache can exceed the 5m default (§2.2). |
| `<think>` blocks in output | Thinking is on by default. Strip server-side (see `thinkFilter` in §6.1), set `reasoning_effort: low`, or append `/no_think` (§3.3). |
| Model "forgets" files mid-task | Context truncation. Confirm the server side (§2.6) **and** that the client isn't sending a smaller `num_ctx` (§2.3, §5.2). |
| Agent never edits files | Tool calling disabled client-side. `"tools": true` in opencode (§5.1). |
| Image input rejected | Your GGUF is a text-only conversion with no `mmproj` projector. Use a repo tagged `image-text-to-text`, or MLX (§0.3). |
| Output quality worse than expected | Wrong sampling for the mode — thinking and non-thinking need different settings (§3.3). Or you pulled an "abliterated"/"uncensored" finetune rather than a plain conversion (§3.1). |
| IP changes every few days | DHCP reservation in your router, or Tailscale for a stable address (§4.6). |

### 7.3 Useful commands

```bash
# Service
launchctl print gui/$(id -u)/local.ollama.serve      # state, exit code, environment
launchctl kickstart -k gui/$(id -u)/local.ollama.serve   # restart after a config change
tail -f ~/Library/Logs/ollama.log                    # the log you configured in §2.4

# What the server actually thinks its config is
grep -a 'server config' ~/Library/Logs/ollama.log | tail -1 | tr ' ' '\n' | grep OLLAMA_

# Models
ollama ps                                            # what's loaded, GPU or CPU
curl -s localhost:11434/api/ps | jq                  # same, plus real context_length
ollama list                                          # what's on disk
ollama show qwen38-code --parameters                 # effective per-model parameters
ollama rm <tag>                                      # reclaim 16 GB

# Host
lsof -nP -iTCP:11434 -sTCP:LISTEN                    # who holds the port
sysctl iogpu.wired_limit_mb                          # GPU wired memory cap
pmset -g custom                                      # sleep settings
```

---

## 8. Open items to verify yourself

Written without direct access to `ollama.com` (blocked by a proxy on the authoring machine), so two
things are unconfirmed:

1. **The official Ollama library tag.** Check <https://ollama.com/library/qwen3.8>. The
   `hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_K_M` fallback in §3.1 works either way.
2. **Whether your Ollama version supports the architecture.** Run the smoke test in §3.1 before
   investing in the rest of the setup. `llama.cpp` is the fallback (§0.3).

Quant sizes in §0.1 and tok/s figures in §0.2 are calculated from parameter count and memory
bandwidth, not measured on your hardware. Treat them as estimates; `ollama ps` and a stopwatch are
authoritative.

Two behaviours worth confirming on your own machine rather than trusting this document:

3. **GPU access from a LaunchDaemon** (§2.7 Option B). Check `ollama ps` reports `100% GPU`; if not,
   fall back to the LaunchAgent plus automatic login.
4. **Exact defaults and variable names for your Ollama version** (§2.2). `ollama serve --help` is
   authoritative for what your build recognises, and the `server config` log line for what it resolved.
