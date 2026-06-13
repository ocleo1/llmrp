# LLM Reverse Proxy

## Usage

```sh
python3.12 -m venv --prompt llmrp .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

## Testing

Run all tests:

```sh
python3 -m pytest -q
```

Run a specific test module:

```sh
python3 -m pytest -q tests/test_integration.py
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

### Linux (systemd)

Add following to `~/.aliases`

```sh
# llmrp systemd
alias llmrp-systemd-install='sudo cp "/path/to/llmrp/llmrp.service" "/etc/systemd/system/llmrp.service" && sudo chown root:root "/etc/systemd/system/llmrp.service" && sudo chmod 644 "/etc/systemd/system/llmrp.service" && sudo systemctl daemon-reload && sudo systemctl enable llmrp.service'
alias llmrp-systemd-uninstall='sudo systemctl disable --now llmrp.service 2>/dev/null; sudo rm -f "/etc/systemd/system/llmrp.service"; sudo systemctl daemon-reload'
alias llmrp-systemd-start='sudo systemctl start llmrp.service'
alias llmrp-systemd-stop='sudo systemctl stop llmrp.service'
alias llmrp-systemd-restart='sudo systemctl restart llmrp.service'
alias llmrp-systemd-status='sudo systemctl status llmrp.service --no-pager'
alias llmrp-systemd-logs='sudo journalctl -u llmrp.service -f'
```

Before installing, update `/path/to/llmrp` and `User`/`Group` in `llmrp.service`.

Run `llmrp-systemd-install` and `llmrp-systemd-start`, it will auto start as a systemd service
