# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tests.test_header

from dimos.agents.agent_ctransformers_gguf import CTransformersGGUFAgent
from dimos.stream.data_provider import QueryDataProvider

# Initialize query stream
query_provider = QueryDataProvider()

# Initialize agent
agent_observable = CTransformersGGUFAgent(
    dev_name="GGUF-Agent",
    model_name="TheBloke/Llama-2-7B-GGUF",
    model_file="llama-2-7b.Q4_K_M.gguf",
    model_type="llama",
    gpu_layers=50,
    max_input_tokens_per_request=250,
    max_output_tokens_per_request=10,
    input_query_stream=query_provider.data_stream
).get_response_observable()

# Subscribe to the observable query stream
agent_observable.subscribe(
    on_next=lambda response: print(f"Response: {response}"),
    on_error=lambda error: print(f"Error: {error}"),
    on_completed=lambda: print("Completed.")
)

# Start the query stream.
# Queries will be pushed every 1 second, in a count from 1 to 5000.
query_provider.start_query_stream(
    query_template=
    "{query}; Denote the number at the beginning of this query before the semicolon as the 'reference number'. Provide the reference number, without any other text in your response.",
    frequency=1,
    start_count=1,
    end_count=5000,
    step=1)

try:
    input("Press ESC to exit...")
except KeyboardInterrupt:
    print("\nExiting...")