from abc import ABC, abstractmethod
import logging
from exceptions.agent_memory_exceptions import UnknownConnectionTypeError, AgentMemoryConnectionError

class AbstractAgentMemory(ABC):
    def __init__(self, connection_type='local', **kwargs):
        """
        Initialize with dynamic connection parameters.
        Args:
            connection_type (str): 'local' for a local database, 'remote' for a remote connection.
        Raises:
            UnknownConnectionTypeError: If an unrecognized connection type is specified.
            AgentMemoryConnectionError: If initializing the database connection fails.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Initializing AgentMemory with connection type: %s', connection_type)
        self.connection_params = kwargs
        self.db_connection = None  # Holds the conection, whether local or remote, to the database used.
        
        if connection_type not in ['local', 'remote']:
            error = UnknownConnectionTypeError(f"Invalid connection_type {connection_type}. Expected 'local' or 'remote'.")
            self.logger.error(str(error))
            raise error

        try:
            if connection_type == 'remote':
                self.connect()
            elif connection_type == 'local':
                self.create()
        except Exception as e:
            self.logger.error("Failed to initialize database connection: %s", str(e), exc_info=True)
            raise AgentMemoryConnectionError("Initialization failed due to an unexpected error.", cause=e) from e

    @abstractmethod
    def connect(self):
        """Establish a connection to the database using dynamic parameters specified during initialization."""

    @abstractmethod
    def create(self):
        """Create a local instance of the database tailored to specific requirements."""

    @abstractmethod
    def add_vector(self, vector_id, vector_data):
        """Add a vector to the database.
        Args:
            vector_id (any): Unique identifier for the vector.
            vector_data (any): The actual data of the vector to be stored.
        """

    @abstractmethod
    def get_vector(self, vector_id):
        """Retrieve a vector from the database by its identifier.
        Args:
            vector_id (any): The identifier of the vector to retrieve.
        """

    @abstractmethod
    def update_vector(self, vector_id, new_vector_data):
        """Update an existing vector in the database.
        Args:
            vector_id (any): The identifier of the vector to update.
            new_vector_data (any): The new data to replace the existing vector data.
        """

    @abstractmethod
    def delete_vector(self, vector_id):
        """Delete a vector from the database using its identifier.
        Args:
            vector_id (any): The identifier of the vector to delete.
        """
