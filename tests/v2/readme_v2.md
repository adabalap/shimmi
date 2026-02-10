# Spock Bot Test Suite v2.0

## 🎯 What's New in v2.0

✅ **Separated data from code** - Test scenarios in JSON file  
✅ **Command-line configurable delays** - Control rate limiting  
✅ **New test phases** - 8+ new scenario categories  
✅ **Flexible configuration** - Use config files or CLI args  
✅ **Better output modes** - Verbose, normal, and quiet modes  
✅ **Enhanced reporting** - Results grouped by phase  

---

## 📁 File Structure

```
.
├── spock_bot_tester_v2.py    # Main test script
├── test_scenarios.json        # Test data (separated from code)
├── config.json                # Configuration file (optional)
└── test_results.json          # Output (auto-generated)
```

---

## 🚀 Quick Start

### 1. Basic Setup

```bash
# Download files
# - spock_bot_tester_v2.py
# - test_scenarios.json
# - config.json (optional)

# Make executable
chmod +x spock_bot_tester_v2.py

# Install dependencies
pip install requests
```

### 2. Edit Configuration

**Option A: Edit config.json**
```json
{
  "webhook_url": "http://YOUR_IP:6000/webhook",
  "delay": 2,
  "message_delay": 10
}
```

**Option B: Use command-line arguments** (overrides config file)

### 3. Run Tests

```bash
# Quick test (phases 1 & 6)
python3 spock_bot_tester_v2.py --quick

# Full standard test (phases 1-10 + bonus)
python3 spock_bot_tester_v2.py --full

# Extended test (includes all new phases)
python3 spock_bot_tester_v2.py --extended
```

---

## 📋 Command Reference

### Test Modes

| Command | Description | Phases Run |
|---------|-------------|------------|
| `--quick` | Quick test | phase_1, phase_6 (~25 msgs) |
| `--full` | Standard full test | phase_1 to phase_10 + bonus (~50 msgs) |
| `--extended` | All phases including new ones | All phases (~100+ msgs) |
| `--phases <list>` | Specific phases | Custom selection |
| `--list` | List all available phases | None (just lists) |

### Configuration Options

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--config FILE` | Config file path | None | `--config my_config.json` |
| `--url URL` | Webhook URL | From config | `--url http://localhost:5000/webhook` |
| `--delay N` | Base delay (seconds) | 2 | `--delay 1` |
| `--msg-delay N` | Extra delay per message | 10 | `--msg-delay 5` |
| `--timeout N` | Request timeout | 30 | `--timeout 60` |

### Data & Output Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--data FILE` | Test scenarios file | `test_scenarios.json` |
| `--output FILE` | Results output file | `test_results.json` |
| `--verbose` | Detailed output | False |
| `--quiet` | Minimal output | False |

---

## 💡 Usage Examples

### Example 1: Quick Development Test
```bash
# Fast iteration with reduced delays
python3 spock_bot_tester_v2.py --quick --delay 0.5 --msg-delay 2
```

### Example 2: Production Rate-Limited Test
```bash
# Respect strict rate limits (12 seconds per message)
python3 spock_bot_tester_v2.py --full --delay 2 --msg-delay 10
```

### Example 3: Test Specific New Phases
```bash
# Test only new relationship and health phases
python3 spock_bot_tester_v2.py --phases new_phase_11 new_phase_14 --verbose
```

### Example 4: Custom Configuration
```bash
# Use custom config and data files
python3 spock_bot_tester_v2.py \
  --config prod_config.json \
  --data custom_scenarios.json \
  --output prod_results_$(date +%Y%m%d).json
```

### Example 5: Memory-Focused Test
```bash
# Test memory phases only
python3 spock_bot_tester_v2.py \
  --phases phase_1 phase_2 phase_3 phase_6 recall_phase_new \
  --msg-delay 8
```

### Example 6: Full Extended Test
```bash
# Complete test with all phases (will take ~20+ minutes)
python3 spock_bot_tester_v2.py --extended --msg-delay 15 --quiet
```

---

## 📊 New Test Phases

### Standard Phases (1-10 + bonus)
Original test suite covering basic functionality, memory, and edge cases.

### New Extended Phases

| Phase | Description | Messages |
|-------|-------------|----------|
| **new_phase_11** | Family & Relationships | 4 |
| **new_phase_12** | Daily Routines & Habits | 4 |
| **new_phase_13** | Financial & Life Goals | 4 |
| **new_phase_14** | Health & Wellness | 4 |
| **new_phase_15** | Entertainment & Media | 4 |
| **recall_phase_new** | Extended recall for new info | 8 |
| **conflict_phase** | Conflicting information handling | 4 |
| **multi_entity_phase** | Multiple similar entities | 4 |
| **privacy_phase** | Sensitive information handling | 4 |

### List All Available Phases
```bash
python3 spock_bot_tester_v2.py --list
```

---

## ⚙️ Configuration Deep Dive

### Understanding Delays

The script uses TWO delay mechanisms:

1. **Base Delay (`--delay`)**: Sleep time built into the request loop (2s default)
2. **Message Delay (`--msg-delay`)**: Additional sleep after each message (10s default)

**Total delay per message** = `delay` + `msg-delay`

#### Why Two Delays?

- **Base delay**: Ensures stable request handling
- **Message delay**: Prevents hitting LLM API rate limits

#### Rate Limit Examples

```bash
# Conservative (15s per message) - 240 msgs/hour
--delay 2 --msg-delay 13

# Standard (12s per message) - 300 msgs/hour  
--delay 2 --msg-delay 10

# Aggressive (7s per message) - 514 msgs/hour
--delay 2 --msg-delay 5

# Development (3s per message) - 1200 msgs/hour
--delay 1 --msg-delay 2
```

### Config File vs CLI Arguments

**Config file** (`config.json`):
- Persistent settings
- Easier to manage
- Can be version controlled

**CLI arguments**:
- Override config file
- Quick one-off changes
- Good for testing different settings

**Priority**: CLI args > Config file > Default values

---

## 📝 Creating Custom Test Scenarios

### Add New Test Phase

Edit `test_scenarios.json`:

```json
{
  "my_custom_phase": {
    "name": "My Custom Tests",
    "description": "Testing my specific features",
    "messages": [
      "Spock test message 1",
      "Spock test message 2",
      "Spock test message 3"
    ]
  }
}
```

Run it:
```bash
python3 spock_bot_tester_v2.py --phases my_custom_phase
```

### Phase with Subsections

For phases that need organization:

```json
{
  "complex_phase": {
    "name": "Complex Multi-Part Test",
    "description": "Testing with subsections",
    "subsections": {
      "part_a": {
        "name": "Part A - Setup",
        "messages": ["Spock setup message"]
      },
      "part_b": {
        "name": "Part B - Testing",
        "messages": ["Spock test message"]
      }
    }
  }
}
```

---

## 📈 Understanding Test Results

### Console Output

**Verbose mode** (`--verbose`):
- Shows full request/response details
- Best for debugging

**Normal mode** (default):
- Progress indicators
- Summary at the end

**Quiet mode** (`--quiet`):
- Minimal output
- Just success/failure icons

### JSON Results File

Structure:
```json
{
  "timestamp": "2026-01-23T14:30:00",
  "config": { ... },
  "total_messages": 50,
  "successful": 48,
  "failed": 2,
  "results": [
    {
      "message_num": 1,
      "phase": "Phase 1",
      "message": "Spock hi there",
      "full_message": "Spock hi there",
      "status_code": 200,
      "success": true,
      "response": "...",
      "full_response": "...",
      "timestamp": "2026-01-23T14:30:05"
    }
  ]
}
```

### Analyzing Results

```bash
# View summary
cat test_results.json | jq '{total_messages, successful, failed}'

# See all failed messages
cat test_results.json | jq '.results[] | select(.success == false)'

# Count by phase
cat test_results.json | jq '.results | group_by(.phase) | map({phase: .[0].phase, count: length})'
```

---

## 🔧 Troubleshooting

### Connection Errors

```
❌ Error: HTTPConnectionPool... Max retries exceeded
```

**Solutions**:
1. Check webhook is running: `curl http://YOUR_IP:6000/webhook`
2. Verify IP and port in config
3. Check firewall rules

### Rate Limit Errors

```
Status: 429
Response: Rate limit exceeded
```

**Solutions**:
1. Increase delays: `--msg-delay 15`
2. Run fewer phases: `--quick` instead of `--full`
3. Use `--phases` to test incrementally

### Timeout Errors

```
❌ Error: ReadTimeout...
```

**Solutions**:
1. Increase timeout: `--timeout 60`
2. Check bot response time
3. Verify LLM API is responding

### Test Data Not Found

```
❌ Test data file not found: test_scenarios.json
```

**Solutions**:
1. Ensure `test_scenarios.json` is in same directory
2. Use `--data /full/path/to/scenarios.json`
3. Check file permissions

---

## 📚 Best Practices

### 1. Start Small, Scale Up
```bash
# Day 1: Quick test
python3 spock_bot_tester_v2.py --quick --msg-delay 5

# Day 2: Add phases gradually
python3 spock_bot_tester_v2.py --phases phase_1 phase_2 phase_6

# Day 3: Full test
python3 spock_bot_tester_v2.py --full
```

### 2. Monitor Your Bot
- Watch bot logs during tests
- Check memory usage
- Verify state persistence

### 3. Version Your Test Data
```bash
# Save different test sets
test_scenarios_v1.json
test_scenarios_v2.json
test_scenarios_production.json
```

### 4. Automated Testing
```bash
#!/bin/bash
# daily_test.sh

DATE=$(date +%Y%m%d)
python3 spock_bot_tester_v2.py \
  --full \
  --msg-delay 12 \
  --output "results/test_${DATE}.json" \
  --quiet

# Check if test passed
if [ $? -eq 0 ]; then
  echo "✅ Tests passed"
else
  echo "❌ Tests failed - check results/test_${DATE}.json"
fi
```

### 5. Test in Different Environments
```bash
# Development
python3 spock_bot_tester_v2.py --config dev_config.json --quick

# Staging
python3 spock_bot_tester_v2.py --config staging_config.json --full

# Production (careful!)
python3 spock_bot_tester_v2.py --config prod_config.json --phases phase_1 phase_6
```

---

## 🎓 Advanced Usage

### Parallel Testing (Multiple Users)

Create multiple configs:
```bash
# user1_config.json
{"user_phone": "111@lid", "user_name": "User1"}

# user2_config.json  
{"user_phone": "222@lid", "user_name": "User2"}

# Run in parallel
python3 spock_bot_tester_v2.py --config user1_config.json --quick &
python3 spock_bot_tester_v2.py --config user2_config.json --quick &
wait
```

### Custom Test Scenarios for Features

```json
{
  "feature_x_test": {
    "name": "Feature X Complete Test",
    "description": "End-to-end test for Feature X",
    "messages": [
      "Spock enable feature X",
      "Spock configure feature X with option A",
      "Spock test feature X behavior",
      "Spock verify feature X results",
      "Spock disable feature X"
    ]
  }
}
```

### Integration with CI/CD

```yaml
# .github/workflows/test-bot.yml
name: Test Spock Bot
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          python3 spock_bot_tester_v2.py \
            --url ${{ secrets.WEBHOOK_URL }} \
            --quick \
            --output test_results.json
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test_results.json
```

---

## 🆘 Getting Help

1. **List available phases**: `python3 spock_bot_tester_v2.py --list`
2. **Check version**: `python3 spock_bot_tester_v2.py --help`
3. **Verbose debugging**: `python3 spock_bot_tester_v2.py --quick --verbose`
4. **Test single message**: Edit `test_scenarios.json` to include only one message

---

## 📄 License & Credits

This test suite is designed for testing the Spock WhatsApp AI Assistant bot.

**Version**: 2.0  
**Last Updated**: January 2026
