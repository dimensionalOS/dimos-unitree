import requests
from dimos.skills.skills import AbstractSkill
from typing import Optional, Dict, Any, List, Tuple
from pydantic import Field
import logging

logger = logging.getLogger(__name__)

class GenericRestSkill(AbstractSkill):
    """Performs a configurable REST API call.

    This skill executes an HTTP request based on the provided parameters. It
    supports various HTTP methods and allows specifying URL, timeout.

    Attributes:
        url: The target URL for the API call.
        method: The HTTP method (e.g., 'GET', 'POST'). Case-insensitive.
        timeout: Request timeout in seconds.
    """ 
    # TODO: Add query parameters, request body data (form-encoded or JSON), and headers.
    #, query
    # parameters, request body data (form-encoded or JSON), and headers.
    # params: Optional dictionary of URL query parameters.
    # data: Optional dictionary for form-encoded request body data.
    # json_payload: Optional dictionary for JSON request body data. Use the
    #     alias 'json' when initializing.
    # headers: Optional dictionary of HTTP headers.
    url: str = Field(..., description="The target URL for the API call.")
    method: str = Field(..., description="HTTP method (e.g., 'GET', 'POST').")
    timeout: int = Field(..., description="Request timeout in seconds.")
    # params: Optional[Dict[str, Any]] = Field(default=None, description="URL query parameters.")
    # data: Optional[Dict[str, Any]] = Field(default=None, description="Form-encoded request body.")
    # json_payload: Optional[Dict[str, Any]] = Field(default=None, alias="json", description="JSON request body.")
    # headers: Optional[Dict[str, str]] = Field(default=None, description="HTTP headers.")


    def __call__(self) -> str:
        """Executes the configured REST API call.

        Returns:
            The text content of the response on success (HTTP 2xx).

        Raises:
            requests.exceptions.RequestException: If a connection error, timeout,
                or other request-related issue occurs.
            requests.exceptions.HTTPError: If the server returns an HTTP 4xx or
                5xx status code.
            Exception: For any other unexpected errors during execution.

        Returns:
             A string representing the success or failure outcome. If successful,
             returns the response body text. If an error occurs, returns a
             descriptive error message.
        """
        try:
            logger.debug(
                f"Executing {self.method.upper()} request to {self.url} "
                f"with timeout={self.timeout}" # , params={self.params}, "
                # f"data={self.data}, json={self.json_payload}, headers={self.headers}"
            )
            response = requests.request(
                method=self.method.upper(), # Normalize method to uppercase
                url=self.url,
                # params=self.params,
                # data=self.data,
                # json=self.json_payload, # Use the attribute name defined in Pydantic
                # headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            logger.debug(f"Request successful. Status: {response.status_code}, Response: {response.text[:100]}...")
            return response.text # Return text content directly
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err} - Status Code: {http_err.response.status_code}")
            return f"HTTP error making {self.method.upper()} request to {self.url}: {http_err.response.status_code} {http_err.response.reason}"
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request exception occurred: {req_err}")
            return f"Error making {self.method.upper()} request to {self.url}: {req_err}"
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}") # Log the full traceback
            return f"An unexpected error occurred: {type(e).__name__}: {e}"


def _control_lights(action: str, urls: List[str], timeout: int) -> Tuple[str, List[str], List[str]]:
    """Helper function to perform an action (on/off) on multiple lights."""
    results = []
    raw_results = []
    success_messages = []

    for i, url in enumerate(urls):
        light_num = i + 1
        try:
            # Use GET method as per original implementation, assuming state change happens via GET
            # Consider changing method if API requires POST/PUT for state changes
            light_skill = GenericRestSkill(url=url, method="GET", timeout=timeout)
            result = light_skill()
            raw_results.append(f"Light {light_num} ({url}): {result}")

            # Check for common error indicators in the response text.
            # This might need adjustment based on the actual API response format.
            if "error" not in result.lower() and "fail" not in result.lower():
                 success_msg = f"Light {light_num} turned {action} successfully."
                 results.append(success_msg)
                 success_messages.append(success_msg)
            else:
                 results.append(f"Light {light_num} failed: {result}")
        except Exception as e:
             # Catch errors during skill instantiation or execution
             error_msg = f"Failed to create/execute request for Light {light_num} ({url}): {type(e).__name__}: {e}"
             raw_results.append(f"Light {light_num} ({url}): {error_msg}")
             results.append(f"Light {light_num} failed: {error_msg}")
             logger.error(error_msg)

    # Aggregate results
    successful_calls = len(success_messages)
    total_lights = len(urls)

    if successful_calls == total_lights:
        final_result = f"Successfully turned {action} all {total_lights} lights."
    elif successful_calls > 0:
        final_result = f"Partially turned {action} lights ({successful_calls}/{total_lights} succeeded). Details: {'; '.join(raw_results)}"
    else:
        final_result = f"Failed to turn {action} any lights. Details: {'; '.join(raw_results)}"

    return final_result, results, raw_results


class TurnOnLight(AbstractSkill):
    """Turns on specific lights via REST API calls.

    This skill attempts to turn on two predefined lights by sending GET requests
    to their respective URLs using the GenericRestSkill.

    Attributes:
        timeout: Request timeout in seconds for each light.
    """
    def __init__(self, timeout: int = 5):
        # TODO: Make these URLs configurable (e.g., via environment variables or config file).
        self._light_urls = [
            "http://0.0.0.0:5123/tasks/1",
            "http://0.0.0.0:5123/tasks/2"
        ]
        self._timeout = timeout
        logger.info(f"TurnOnLight skill initialized for URLs: {self._light_urls} with timeout {timeout}s.")

    def __call__(self) -> str:
        """Executes the sequence to turn on the lights.

        Returns:
            A string summarizing the outcome (success, partial success, failure)
            including details from the raw API responses in case of issues.
        """
        final_result, _, raw_results = _control_lights(
            action="on",
            urls=self._light_urls,
            timeout=self._timeout
        )
        logger.info(f"TurnOnLight result: {final_result}")
        # Log raw results only if there was a partial success or failure for brevity
        if "Partially" in final_result or "Failed" in final_result:
            logger.debug(f"TurnOnLight raw results: {'; '.join(raw_results)}")
        return final_result


class TurnOffLight(AbstractSkill):
    """Turns off specific lights via REST API calls.

    This skill attempts to turn off two predefined lights by sending GET requests
    to their respective URLs using the GenericRestSkill.

    Attributes:
        timeout: Request timeout in seconds for each light.
    """
    def __init__(self, timeout: int = 5):
        # TODO: Make these URLs configurable (e.g., via environment variables or config file).
        self._light_urls = [
            "http://0.0.0.0:5123/tasks/1",
            "http://0.0.0.0:5123/tasks/2"
        ]
        self._timeout = timeout
        logger.info(f"TurnOffLight skill initialized for URLs: {self._light_urls} with timeout {timeout}s.")


    def __call__(self) -> str:
        """Executes the sequence to turn off the lights.

        Returns:
            A string summarizing the outcome (success, partial success, failure)
            including details from the raw API responses in case of issues.
        """
        final_result, _, raw_results = _control_lights(
            action="off",
            urls=self._light_urls,
            timeout=self._timeout
        )
        logger.info(f"TurnOffLight result: {final_result}")
        # Log raw results only if there was a partial success or failure for brevity
        if "Partially" in final_result or "Failed" in final_result:
            logger.debug(f"TurnOffLight raw results: {'; '.join(raw_results)}")
        return final_result