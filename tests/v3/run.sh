#python3 spock_bot_tester_v3.py \
#  --config config.json \
#  --data test_scenarios_v3.json \
#  --quick \
#  --output test_results_v3.json


python3 spock_bot_tester_v3.py \
  --config config.json \
  --data test_scenarios_v3.json \
  --phases phase_1_memory_basics phase_3_privacy_security \
  --output results_privacy.json

#python3 spock_bot_tester_v3.py --data test_scenarios_v3.json --list
