#!/usr/bin/env python3
"""
Spock Bot Test Script v2.0
Sends test payloads to your Flask webhook to test memory and state persistence
Supports external test scenario files and configurable parameters
"""

import requests
import json
import time
import argparse
from datetime import datetime
import uuid
import sys
import os
from pathlib import Path

# Default Configuration
DEFAULT_CONFIG = {
    "webhook_url": "http://129.159.226.51:6000/webhook",
    "user_phone": "4930656034916@lid",
    "user_name": "Sarah TestUser",
    "bot_phone": "919573717667@c.us",
    "session": "default",
    "delay": 2,
    "message_delay": 10,  # Additional delay after each message (for rate limiting)
    "timeout": 30
}


class SpockTester:
    def __init__(self, config, test_data):
        self.config = config
        self.test_data = test_data
        self.message_count = 0
        self.test_results = []
        
    def generate_message_id(self):
        """Generate a unique message ID similar to WhatsApp format"""
        timestamp = int(time.time())
        unique = str(uuid.uuid4())[:8].upper()
        return f"true_{self.config['user_phone'].split('@')[0]}_{timestamp}_{unique}"
    
    def create_payload(self, message_body):
        """Create WAHA webhook payload"""
        timestamp = int(time.time())
        message_id = self.generate_message_id()
        
        payload = {
            "id": f"evt_{uuid.uuid4().hex}",
            "session": self.config['session'],
            "event": "message.any",
            "payload": {
                "id": message_id,
                "timestamp": timestamp,
                "from": self.config['user_phone'],
                "fromMe": True,
                "source": "app",
                "body": message_body,
                "hasMedia": False,
                "media": None,
                "ack": 1,
                "ackName": "SERVER",
                "location": None,
                "vCards": None,
                "replyTo": None,
                "_data": {
                    "key": {
                        "remoteJid": self.config['user_phone'],
                        "fromMe": True,
                        "id": message_id
                    },
                    "messageTimestamp": timestamp,
                    "pushName": self.config['user_name'],
                    "broadcast": False,
                    "status": 2,
                    "message": {
                        "conversation": message_body
                    }
                }
            },
            "timestamp": timestamp * 1000,
            "metadata": {},
            "me": {
                "id": self.config['bot_phone'],
                "pushName": "Spock Bot",
                "lid": self.config['user_phone']
            },
            "engine": "NOWEB",
            "environment": {
                "version": "2025.11.3",
                "engine": "NOWEB",
                "tier": "CORE",
                "browser": None
            }
        }
        return payload
    
    def send_message(self, message, phase_name="", test_description="", verbose=True):
        """Send a message to the webhook"""
        self.message_count += 1
        payload = self.create_payload(message)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Message #{self.message_count}")
            if phase_name:
                print(f"Phase: {phase_name}")
            if test_description:
                print(f"Test: {test_description}")
            print(f"{'='*80}")
            print(f"Sending: {message[:100]}{'...' if len(message) > 100 else ''}")
        
        try:
            response = requests.post(
                self.config['webhook_url'],
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=self.config['timeout']
            )
            
            result = {
                'message_num': self.message_count,
                'phase': phase_name,
                'test': test_description,
                'message': message[:50] + '...' if len(message) > 50 else message,
                'full_message': message,
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response': response.text[:500] if response.text else 'No response',
                'full_response': response.text,
                'timestamp': datetime.now().isoformat()
            }
            
            if verbose:
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text[:200]}")
            else:
                status_icon = "✅" if response.status_code == 200 else "❌"
                print(f"{status_icon} Msg #{self.message_count}: {message[:50]}...")
            
            self.test_results.append(result)
            
            # Wait before next message (base delay + message delay for rate limiting)
            total_delay = self.config['delay'] + self.config['message_delay']
            if verbose and total_delay > 0:
                print(f"⏳ Waiting {total_delay}s before next message...")
            time.sleep(total_delay)
            
            return response
            
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"❌ Error: {e}")
            else:
                print(f"❌ Msg #{self.message_count}: Error - {str(e)[:50]}")
            
            result = {
                'message_num': self.message_count,
                'phase': phase_name,
                'test': test_description,
                'message': message[:50],
                'full_message': message,
                'status_code': 0,
                'success': False,
                'response': str(e),
                'full_response': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.test_results.append(result)
            return None
    
    def run_phase(self, phase_key, verbose=True):
        """Run a single phase from test data"""
        if phase_key not in self.test_data:
            print(f"❌ Phase '{phase_key}' not found in test data")
            return
        
        phase = self.test_data[phase_key]
        
        if verbose:
            print("\n" + "🔷"*40)
            print(f"PHASE: {phase['name']}")
            print(f"Description: {phase['description']}")
            print("🔷"*40)
        else:
            print(f"\n📋 Running Phase: {phase['name']}")
        
        # Handle phases with subsections
        if 'subsections' in phase:
            for subsection_key, subsection in phase['subsections'].items():
                if verbose:
                    print(f"\n--- {subsection['name']} ---")
                for msg in subsection['messages']:
                    self.send_message(
                        msg, 
                        f"{phase['name']} - {subsection['name']}", 
                        phase['description'],
                        verbose
                    )
        # Handle regular phases
        elif 'messages' in phase:
            for msg in phase['messages']:
                self.send_message(msg, phase['name'], phase['description'], verbose)
    
    def run_phases(self, phase_keys, verbose=True):
        """Run multiple phases"""
        for phase_key in phase_keys:
            self.run_phase(phase_key, verbose)
    
    def list_phases(self):
        """List all available phases"""
        print("\n" + "="*80)
        print("AVAILABLE TEST PHASES")
        print("="*80)
        
        for idx, (key, phase) in enumerate(self.test_data.items(), 1):
            msg_count = 0
            if 'subsections' in phase:
                msg_count = sum(len(s['messages']) for s in phase['subsections'].values())
            elif 'messages' in phase:
                msg_count = len(phase['messages'])
            
            print(f"\n{idx}. {key}")
            print(f"   Name: {phase['name']}")
            print(f"   Description: {phase['description']}")
            print(f"   Messages: {msg_count}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        total = len(self.test_results)
        successful = sum(1 for r in self.test_results if r['success'])
        failed = total - successful
        
        print(f"\nTotal Messages Sent: {total}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        if total > 0:
            print(f"Success Rate: {(successful/total*100):.1f}%")
        
        # Group results by phase
        phases = {}
        for result in self.test_results:
            phase = result['phase'] or 'Unknown'
            if phase not in phases:
                phases[phase] = {'total': 0, 'success': 0, 'failed': 0}
            phases[phase]['total'] += 1
            if result['success']:
                phases[phase]['success'] += 1
            else:
                phases[phase]['failed'] += 1
        
        print("\n" + "-"*80)
        print("RESULTS BY PHASE")
        print("-"*80)
        for phase, stats in phases.items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"\n{phase}:")
            print(f"  Total: {stats['total']}, Success: {stats['success']}, "
                  f"Failed: {stats['failed']}, Rate: {success_rate:.1f}%")
        
        if failed > 0:
            print("\n" + "-"*80)
            print("❌ FAILED MESSAGES")
            print("-"*80)
            for result in self.test_results:
                if not result['success']:
                    print(f"\n  Msg #{result['message_num']} [{result['phase']}]")
                    print(f"  Message: {result['message']}")
                    print(f"  Status: {result['status_code']}")
                    print(f"  Error: {result['response'][:100]}")
        
        print("\n" + "="*80)
    
    def save_results(self, filename="test_results.json"):
        """Save test results to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'total_messages': len(self.test_results),
            'successful': sum(1 for r in self.test_results if r['success']),
            'failed': sum(1 for r in self.test_results if not r['success']),
            'results': self.test_results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")


def load_config(config_file):
    """Load configuration from JSON file"""
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            custom_config = json.load(f)
            config = DEFAULT_CONFIG.copy()
            config.update(custom_config)
            return config
    return DEFAULT_CONFIG.copy()


def load_test_data(data_file):
    """Load test scenarios from JSON file"""
    if not Path(data_file).exists():
        print(f"❌ Test data file not found: {data_file}")
        sys.exit(1)
    
    with open(data_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Test Spock WhatsApp Bot - v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test
  python3 %(prog)s --quick
  
  # Run specific phases
  python3 %(prog)s --phases phase_1 phase_6
  
  # Custom configuration
  python3 %(prog)s --url http://localhost:5000/webhook --delay 1 --msg-delay 5
  
  # Use custom test data
  python3 %(prog)s --data my_scenarios.json --phases new_phase_11
  
  # List all available phases
  python3 %(prog)s --list
        """
    )
    
    # Configuration arguments
    parser.add_argument('--config', help='Config file (JSON)', default=None)
    parser.add_argument('--url', help='Webhook URL', default=None)
    parser.add_argument('--delay', type=float, help='Base delay between messages (seconds)', default=None)
    parser.add_argument('--msg-delay', type=float, help='Additional delay after each message (seconds)', default=None)
    parser.add_argument('--timeout', type=int, help='Request timeout (seconds)', default=None)
    
    # Test data arguments
    parser.add_argument('--data', help='Test scenarios file (JSON)', default='test_scenarios.json')
    parser.add_argument('--phases', nargs='+', help='Specific phases to run')
    parser.add_argument('--quick', action='store_true', help='Run quick test (phase_1, phase_6)')
    parser.add_argument('--full', action='store_true', help='Run all standard phases (1-10 + bonus)')
    parser.add_argument('--extended', action='store_true', help='Run all phases including new ones')
    parser.add_argument('--list', action='store_true', help='List all available phases and exit')
    
    # Output arguments
    parser.add_argument('--output', default='test_results.json', help='Output file for results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output (show full details)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.url:
        config['webhook_url'] = args.url
    if args.delay is not None:
        config['delay'] = args.delay
    if args.msg_delay is not None:
        config['message_delay'] = args.msg_delay
    if args.timeout:
        config['timeout'] = args.timeout
    
    # Load test data
    test_data = load_test_data(args.data)
    
    # Initialize tester
    tester = SpockTester(config, test_data)
    
    # Handle --list
    if args.list:
        tester.list_phases()
        return
    
    # Determine verbosity
    verbose = args.verbose and not args.quiet
    
    # Print header
    if not args.quiet:
        print("="*80)
        print("SPOCK BOT COMPREHENSIVE TEST SUITE v2.0")
        print("="*80)
        print(f"Webhook URL: {config['webhook_url']}")
        print(f"Base delay: {config['delay']}s")
        print(f"Message delay: {config['message_delay']}s (for rate limiting)")
        print(f"Total delay per message: {config['delay'] + config['message_delay']}s")
        print(f"Test User: {config['user_name']} ({config['user_phone']})")
        print(f"Test Data: {args.data}")
        print("="*80)
    
    try:
        # Determine which phases to run
        if args.quick:
            if not args.quiet:
                print("\n🚀 Running QUICK TEST mode (phase_1, phase_6)")
            tester.run_phases(['phase_1', 'phase_6'], verbose)
        
        elif args.full:
            if not args.quiet:
                print("\n🚀 Running FULL TEST SUITE (all standard phases)")
            phases = [f'phase_{i}' for i in range(1, 11)] + ['bonus']
            tester.run_phases(phases, verbose)
        
        elif args.extended:
            if not args.quiet:
                print("\n🚀 Running EXTENDED TEST SUITE (all phases)")
            tester.run_phases(list(test_data.keys()), verbose)
        
        elif args.phases:
            if not args.quiet:
                print(f"\n🚀 Running selected phases: {', '.join(args.phases)}")
            tester.run_phases(args.phases, verbose)
        
        else:
            # Default: run standard phases
            if not args.quiet:
                print("\n🚀 Running DEFAULT TEST SUITE (phases 1-10 + bonus)")
            phases = [f'phase_{i}' for i in range(1, 11)] + ['bonus']
            tester.run_phases(phases, verbose)
        
        # Print summary and save results
        tester.print_summary()
        tester.save_results(args.output)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        tester.print_summary()
        tester.save_results(args.output)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        tester.save_results(args.output)


if __name__ == "__main__":
    main()
