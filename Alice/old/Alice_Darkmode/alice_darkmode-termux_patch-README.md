## Needs testing
## intended for alice_darkmode.py -// not tested with 0.0.2.py

0) Prereqs (once)
```
pkg update
pkg install git patch nano
```

1) Save the patch file
```
cd ~/Alice
mkdir -p patches
cat > patches/alice_darkmode-termux.patch <<'PATCH'

PATCH
```

2) Apply the patch
```
cd ~/Alice
git status                                   # sanity check
git checkout -b termux-support || git switch -c termux-support
git apply --check patches/alice_darkmode-termux.patch # dry-run
git apply --index patches/alice_darkmode-termux.patch # apply + stage
git commit -m "Termux support: timeouts, wake-lock, Ollama readiness, etc."
```

**If you are not using git:**
```
cd ~/Alice
patch -p1 < patches/alice_darkmode-termux.patch
```

## Game on
3) Use it (Termux way)
```
termux-wake-lock
```

## Start Ollama locally (background) and wait for its HTTP API:
```
nohup ollama serve >/dev/null 2>&1 &
for i in $(seq 1 30); do
  curl -fsS http://127.0.0.1:11434/api/tags >/dev/null && break
  sleep 0.4
done
```

## Pull the moded if you haven't
```
ollama list | awk '{print $1}' | grep -qx 'llama3.2:3b' || ollama pull llama3.2:3b
```

## Initialize Alice Darkmode
```
cd ~/Alice
python3 alice_darkmode.py \
  --model llama3.2:3b \
  --ollama-host 127.0.0.1 \
  --ollama-port 11434
```
