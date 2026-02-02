#!/usr/bin/env python3
"""
Spock Bot Test Script
Sends test payloads to your Flask webhook to test memory and state persistence
"""

import requests
import json
import time
import argparse
from datetime import datetime
import uuid

# Configuration
WEBHOOK_URL = "http://129.159.226.51:6000/webhook"  # Change this to your webhook URL
USER_PHONE = "4930656034916@lid"
USER_NAME = "Sarah TestUser"
BOT_PHONE = "919573717667@c.us"
SESSION = "default"

class SpockTester:
    def __init__(self, webhook_url, delay=2):
        self.webhook_url = webhook_url
        self.delay = delay
        self.message_count = 0
        self.test_results = []
        
    def generate_message_id(self):
        """Generate a unique message ID similar to WhatsApp format"""
        timestamp = int(time.time())
        unique = str(uuid.uuid4())[:8].upper()
        return f"true_{USER_PHONE.split('@')[0]}_{timestamp}_{unique}"
    
    def create_payload(self, message_body):
        """Create WAHA webhook payload"""
        timestamp = int(time.time())
        message_id = self.generate_message_id()
        
        payload = {
            "id": f"evt_{uuid.uuid4().hex}",
            "session": SESSION,
            "event": "message.any",
            "payload": {
                "id": message_id,
                "timestamp": timestamp,
                "from": USER_PHONE,
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
                        "remoteJid": USER_PHONE,
                        "fromMe": True,
                        "id": message_id
                    },
                    "messageTimestamp": timestamp,
                    "pushName": USER_NAME,
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
                "id": BOT_PHONE,
                "pushName": "Spock Bot",
                "lid": USER_PHONE
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
    
    def send_message(self, message, phase_name="", test_description=""):
        """Send a message to the webhook"""
        self.message_count += 1
        payload = self.create_payload(message)
        
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
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            result = {
                'message_num': self.message_count,
                'phase': phase_name,
                'test': test_description,
                'message': message[:50] + '...' if len(message) > 50 else message,
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response': response.text[:200] if response.text else 'No response'
            }
            
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
            self.test_results.append(result)
            
            # Wait before next message
            time.sleep(self.delay)
            return response
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            result = {
                'message_num': self.message_count,
                'phase': phase_name,
                'test': test_description,
                'message': message[:50],
                'status_code': 0,
                'success': False,
                'response': str(e)
            }
            self.test_results.append(result)
            return None
    
    def run_phase_1(self):
        """Phase 1: Initial Information Gathering (Short Messages)"""
        print("\n" + "🔷"*40)
        print("PHASE 1: Initial Information Gathering")
        print("🔷"*40)
        
        messages = [
            "Spock hi there",
            "Spock my name is Sarah",
            "Spock I'm 28 years old",
            "Spock I live in Seattle",
            "Spock I work as a software engineer"
        ]
        
        for msg in messages:
            self.send_message(msg, "Phase 1", "Short message")
            time.sleep(10)
    
    def run_phase_2(self):
        """Phase 2: Detailed Context Building (Medium Messages)"""
        print("\n" + "🔷"*40)
        print("PHASE 2: Detailed Context Building")
        print("🔷"*40)
        
        messages = [
            "Spock I'm working on a really interesting project right now. It's a machine learning application that helps predict customer churn for e-commerce platforms. We're using Python and TensorFlow for the backend.",
            
            "Spock my hobbies include hiking, photography, and cooking Italian food. I especially love making homemade pasta on weekends. My favorite trail is the Rattlesnake Ledge trail near Seattle.",
            
            "Spock I have two pets - a golden retriever named Max who is 3 years old, and a cat named Luna who is 5. Max loves playing fetch at the park, and Luna is pretty independent but loves cuddles in the evening."
        ]
        
        for msg in messages:
            self.send_message(msg, "Phase 2", "Medium message with context")
            time.sleep(10)
    
    def run_phase_3(self):
        """Phase 3: Complex Narratives (Long Messages)"""
        print("\n" + "🔷"*40)
        print("PHASE 3: Complex Narratives")
        print("🔷"*40)
        
        messages = [
            "Spock let me tell you about my educational background. I graduated from the University of Washington in 2018 with a Bachelor's degree in Computer Science. During my time there, I was particularly interested in artificial intelligence and machine learning courses. I did an internship at Microsoft during my junior year, which was an incredible experience. I worked on the Azure team, specifically on cloud infrastructure optimization. After graduation, I joined a startup called TechVenture as a junior developer. I spent two years there learning full-stack development, working with React, Node.js, and PostgreSQL. The startup culture was intense but rewarding - we were a team of just 12 people trying to build a SaaS platform for project management. Unfortunately, the startup didn't secure Series B funding and had to shut down in 2020. That's when I joined my current company, DataFlow Solutions, where I've been for the past four years working my way up to senior engineer.",
            
            "Spock I should also mention some personal goals and aspirations I have. In the next year, I want to complete a marathon - I've been training for the Seattle Marathon happening in November. I'm currently running about 30 miles per week and following a structured training plan. I also want to learn Spanish, as I'm planning a trip to Spain and Portugal next summer with my best friend Emma, who I've known since college. We're planning to spend three weeks backpacking through Barcelona, Madrid, Lisbon, and Porto. I'm really excited about trying authentic paella and port wine. Career-wise, I'm aiming to transition into a machine learning engineering role within the next two years. I've been taking online courses on Coursera and deeplearning.ai to build up my skills. I'm particularly interested in natural language processing and computer vision applications. My dream would be to work on something that has real social impact, maybe in healthcare or education technology."
        ]
        
        for msg in messages:
            self.send_message(msg, "Phase 3", "Long narrative message")
            time.sleep(10)
    
    def run_phase_4(self):
        """Phase 4: Additional Scattered Details"""
        print("\n" + "🔷"*40)
        print("PHASE 4: Additional Scattered Details")
        print("🔷"*40)
        
        messages = [
            "Spock forgot to mention, my favorite color is teal",
            
            "Spock I'm allergic to peanuts, which can be annoying when eating out",
            
            "Spock my birthday is July 15th, 1996, so I'll be turning 29 this year",
            
            "Spock I drive a 2019 Toyota Prius. I chose it because I wanted something fuel-efficient and reliable for my daily commute. My office is about 25 minutes from my apartment in the Fremont neighborhood. I usually listen to podcasts during my commute - my favorites are Hidden Brain, Radiolab, and The Daily. Sometimes I'll listen to music instead, mostly indie rock and electronic music. My favorite bands are Tame Impala, ODESZA, and The National.",
            
            "Spock my coffee order is a medium oat milk latte with one pump of vanilla"
        ]
        
        for msg in messages:
            self.send_message(msg, "Phase 4", "Scattered details")
            time.sleep(10)
    
    def run_phase_5(self):
        """Phase 5: Unrelated Conversations"""
        print("\n" + "🔷"*40)
        print("PHASE 5: Unrelated Conversations")
        print("🔷"*40)
        
        messages = [
            "Spock what's 245 multiplied by 67?",
            "Spock can you explain what quantum computing is?",
            "Spock tell me a fun fact about space",
            "Spock what's the weather usually like in December?"
        ]
        
        for msg in messages:
            self.send_message(msg, "Phase 5", "Context switching")
            time.sleep(10)
    
    def run_phase_6(self):
        """Phase 6: Memory Recall Tests"""
        print("\n" + "🔷"*40)
        print("PHASE 6: Memory Recall Tests (CRITICAL)")
        print("🔷"*40)
        
        # Immediate Recall
        print("\n--- Immediate Recall ---")
        immediate = [
            "Spock what's my coffee order?",
            "Spock what podcasts do I listen to?",
            "Spock where am I planning to travel next summer?"
        ]
        for msg in immediate:
            self.send_message(msg, "Phase 6 - Immediate", "Recent memory recall")
            time.sleep(10)
        
        # Medium-term Recall
        print("\n--- Medium-term Recall ---")
        medium = [
            "Spock what are my pets' names and ages?",
            "Spock what's my favorite hiking trail?",
            "Spock what project am I currently working on?"
        ]
        for msg in medium:
            self.send_message(msg, "Phase 6 - Medium", "Mid-conversation recall")
            time.sleep(10)
        
        # Long-term Recall
        print("\n--- Long-term Recall ---")
        longterm = [
            "Spock what's my name and age?",
            "Spock where do I live and what do I do for work?",
            "Spock which university did I graduate from and when?"
        ]
        for msg in longterm:
            self.send_message(msg, "Phase 6 - Long-term", "Early conversation recall")
            time.sleep(10)
        
        # Complex Recall
        print("\n--- Complex Recall ---")
        complex_recall = [
            "Spock can you summarize my career journey from college to now?",
            "Spock what are all my hobbies and interests you know about?",
            "Spock tell me about my educational background and career goals"
        ]
        for msg in complex_recall:
            self.send_message(msg, "Phase 6 - Complex", "Synthesis recall")
            time.sleep(10)
        
        # Specific Details
        print("\n--- Specific Detail Recall ---")
        details = [
            "Spock what's my birthday?",
            "Spock what year and model is my car?",
            "Spock what am I allergic to?",
            "Spock what's my favorite color?",
            "Spock how many miles per week am I running?",
            "Spock what startup did I work at and what happened to it?",
            "Spock who is Emma and what's our plan together?"
        ]
        for msg in details:
            self.send_message(msg, "Phase 6 - Details", "Specific fact recall")
            time.sleep(10)
    
    def run_phase_7(self):
        """Phase 7: Update & Override Tests"""
        print("\n" + "🔷"*40)
        print("PHASE 7: Update & Override Tests")
        print("🔷"*40)
        
        updates = [
            "Spock actually, I just got a promotion! I'm now a lead engineer at DataFlow Solutions",
            "Spock I adopted another pet - a rescue cat named Oliver who is 2 years old",
            "Spock my running mileage has increased to 40 miles per week now"
        ]
        
        for msg in updates:
            self.send_message(msg, "Phase 7", "Information update")
            time.sleep(10)
        
        print("\n--- Verify Updates ---")
        verifications = [
            "Spock what's my current job title?",
            "Spock how many pets do I have and what are their names?",
            "Spock how many miles per week am I running now?"
        ]
        
        for msg in verifications:
            self.send_message(msg, "Phase 7", "Update verification")
            time.sleep(10)
    
    def run_phase_8(self):
        """Phase 8: Cross-referencing Multiple Facts"""
        print("\n" + "🔷"*40)
        print("PHASE 8: Cross-referencing")
        print("🔷"*40)
        
        messages = [
            "Spock based on everything you know about me, what kind of vacation would I enjoy?",
            "Spock considering my career goals and current skills, what should I focus on learning?",
            "Spock what connections can you make between my hobbies and my professional work?"
        ]
        
        for msg in messages:
            self.send_message(msg, "Phase 8", "Multi-fact synthesis")
            time.sleep(10)
    
    def run_phase_9(self):
        """Phase 9: Temporal Information"""
        print("\n" + "🔷"*40)
        print("PHASE 9: Temporal Information")
        print("🔷"*40)
        
        messages = [
            "Spock I have a meeting next Tuesday at 2pm with the product team about the ML project",
            "Spock I'm planning to visit my parents in Portland on the weekend of March 15th",
            "Spock remind me what I told you about my schedule"
        ]
        
        for msg in messages:
            self.send_message(msg, "Phase 9", "Temporal memory")
            time.sleep(10)
    
    def run_phase_10(self):
        """Phase 10: Final Comprehensive Test"""
        print("\n" + "🔷"*40)
        print("PHASE 10: Final Comprehensive Test")
        print("🔷"*40)
        
        self.send_message(
            "Spock tell me everything you remember about me - my personal info, career, hobbies, pets, goals, and anything else we've discussed",
            "Phase 10",
            "Complete memory dump"
        )
    
    def run_bonus_tests(self):
        """Bonus Edge Cases"""
        print("\n" + "🔷"*40)
        print("BONUS: Edge Cases")
        print("🔷"*40)
        
        messages = [
            "Spock I never told you this, but what's my shoe size?",
            "Spock did I mention my favorite food?",
            "Spock how long have we been talking?"
        ]
        
        for msg in messages:
            self.send_message(msg, "Bonus", "Edge case testing")
            time.sleep(10)
    
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
        print(f"Success Rate: {(successful/total*100):.1f}%")
        
        if failed > 0:
            print("\n❌ Failed Messages:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - Msg #{result['message_num']}: {result['message']}")
                    print(f"    Status: {result['status_code']}, Response: {result['response'][:50]}")
        
        print("\n" + "="*80)
    
    def save_results(self, filename="test_results.json"):
        """Save test results to JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_messages': len(self.test_results),
                'results': self.test_results
            }, f, indent=2)
        print(f"\n💾 Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Test Spock WhatsApp Bot')
    parser.add_argument('--url', default=WEBHOOK_URL, help='Webhook URL')
    parser.add_argument('--delay', type=float, default=2, help='Delay between messages (seconds)')
    parser.add_argument('--phases', nargs='+', type=int, help='Specific phases to run (1-10, bonus)')
    parser.add_argument('--quick', action='store_true', help='Run quick test (phases 1, 6 only)')
    parser.add_argument('--output', default='test_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    tester = SpockTester(args.url, args.delay)
    
    print("="*80)
    print("SPOCK BOT COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Webhook URL: {args.url}")
    print(f"Delay between messages: {args.delay}s")
    print(f"Test User: {USER_NAME} ({USER_PHONE})")
    print("="*80)
    
    try:
        if args.quick:
            print("\n🚀 Running QUICK TEST mode (Phases 1 & 6 only)")
            tester.run_phase_1()
            tester.run_phase_6()
        elif args.phases:
            print(f"\n🚀 Running selected phases: {args.phases}")
            for phase in args.phases:
                if phase == 1:
                    tester.run_phase_1()
                elif phase == 2:
                    tester.run_phase_2()
                elif phase == 3:
                    tester.run_phase_3()
                elif phase == 4:
                    tester.run_phase_4()
                elif phase == 5:
                    tester.run_phase_5()
                elif phase == 6:
                    tester.run_phase_6()
                elif phase == 7:
                    tester.run_phase_7()
                elif phase == 8:
                    tester.run_phase_8()
                elif phase == 9:
                    tester.run_phase_9()
                elif phase == 10:
                    tester.run_phase_10()
        else:
            print("\n🚀 Running FULL TEST SUITE (All phases + bonus)")
            tester.run_phase_1()
            tester.run_phase_2()
            tester.run_phase_3()
            tester.run_phase_4()
            tester.run_phase_5()
            tester.run_phase_6()
            tester.run_phase_7()
            tester.run_phase_8()
            tester.run_phase_9()
            tester.run_phase_10()
            tester.run_bonus_tests()
        
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


if __name__ == "__main__":
    main()
