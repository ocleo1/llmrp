# LLM Reverse Proxy

## Usage

```sh
python3.12 -m venv --prompt llmrp .venv
source .venv/bin/active
pip install -r requirements.txt
python server.py
```

## Daemon Mode

### MacOS

Add following to `~/.aliases`

```sh
# llmrp LaunchDaemon
alias llmrp-daemon-install='sudo cp "/path/to/llmrp/llmrp.daemon.plist" "/Library/LaunchDaemons/llmrp.daemon.plist" && sudo chown root:wheel "/Library/LaunchDaemons/llmrp.daemon.plist" && sudo chmod 644 "/Library/LaunchDaemons/llmrp.daemon.plist"'
alias llmrp-daemon-uninstall='sudo launchctl bootout "system/llmrp.daemon" 2>/dev/null; sudo rm -f "/Library/LaunchDaemons/llmrp.daemon.plist"'
alias llmrp-daemon-start='sudo launchctl bootstrap system "/Library/LaunchDaemons/llmrp.daemon.plist"'
alias llmrp-daemon-stop='sudo launchctl bootout "system/llmrp.daemon"'
alias llmrp-daemon-restart='sudo launchctl kickstart -k "system/llmrp.daemon"'
alias llmrp-daemon-status='sudo launchctl print "system/llmrp.daemon"'
alias llmrp-daemon-logs='tail -f "/path/to/llmrp/stdout.log" "/path/to/llmrp/stderr.log"'
```

Run `llmrp-daemon-install` and `llmrp-daemon-start`, it will auto start as a LaunchDaemon
