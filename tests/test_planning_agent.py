"""Test file demonstrating PlanningAgent with OpenAIAgent integration.

This test shows how to:
1. Create a PlanningAgent for task planning
2. Create an OpenAIAgent that executes the plan steps
3. Connect them using ReactiveX observables
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv
from openai import NOT_GIVEN

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dimos.agents.agent import OpenAIAgent
from dimos.agents.planning_agent import PlanningAgent
from dimos.utils.logging_config import setup_logger

# Load environment variables
load_dotenv()

def main():
    # Set up logging
    logger = setup_logger("test_planning")
    
    # Initialize agents as None for proper scope in finally block
    planner = None
    executor = None
    
    try:
        # Initialize the planning agent
        planner = PlanningAgent(
            dev_name="TaskPlanner",
            model_name="gpt-4",
            max_steps=10
        )
        
        # Initialize the execution agent that will receive plan steps
        executor = OpenAIAgent(
            dev_name="StepExecutor",
            agent_type="Executor",
            input_query_stream=planner.get_plan_observable(),
            system_query="You are an execution agent that receives steps of a plan and provides feedback on their execution.",
            system_query_without_documents="You are an execution agent that receives steps of a plan and provides feedback on their execution."
        )
        
        # Disable skills for the executor since we just want it to process steps
        executor.skills.set_tools(NOT_GIVEN)
        
        # Main interaction loop
        print("\nWelcome to the Planning Assistant!")
        print("Describe your task and I'll help break it down into steps.")
        print("Once you're happy with the plan, type 'yes' to execute it.")
        print("Type 'quit' to exit.\n")
        
        while True:
            # Get user input
            user_input = input("> ")
            
            # Check for quit command
            if user_input.lower() in ['quit', 'exit']:
                break
                
            # Process the input
            planner.process_user_input(user_input)
            
            # Get the response from the planner
            response = planner.response_subject._subject.value
            
            # Display response based on type
            if response["type"] == "dialogue":
                print("\nPlanner:", response["content"])
            elif response["type"] == "plan":
                print("\nProposed Plan:")
                for i, step in enumerate(response["content"], 1):
                    print(f"{i}. {step}")
                print("\nDoes this plan look good? (yes/no)")
                
            # If plan was confirmed, wait for executor responses
            if planner.plan_confirmed:
                print("\nExecuting plan...")
                # Get responses from executor
                executor_response = executor.response_subject._subject.value
                if executor_response:
                    print("Executor:", executor_response)
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        # Clean up
        if planner:
            planner.dispose_all()
        if executor:
            executor.dispose_all()
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 