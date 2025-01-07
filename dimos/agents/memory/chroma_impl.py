from agents.memory.base import AbstractAgentMemory

from langchain_openai import OpenAIEmbeddings


class AgentMemoryChroma(AbstractAgentMemory):
    def __init__(self, connection_type='local', host='localhost', port=6379, db=0):
        """Initialize the connection to the Chroma DB.
        Args:
            host (str): The host on which Chroma DB is running.
            port (int): The port on which Chroma DB is accessible.
            db (int): The database index to use.
            connection_type (str): Whether to connect to a local or remote database.'
        """
        super().__init__(connection_type=connection_type, host=host, port=port, db=db)
        self.db_connection
        

    def connect(self):
        try:
            import dimos.agents.memory.chroma_impl as chroma_impl
            self.connection = chroma_impl.connect(self.host, self.port, self.db)
            self.logger.info("Connected successfully to Chroma DB")
        except Exception as e:
            self.logger.error("Failed to connect to Chroma DB", exc_info=True)

    def add_vector(self, vector_id, vector_data):
        try:
            self.connection.add(vector_id, vector_data)
        except Exception as e:
            self.logger.error(f"Failed to add vector {vector_id}", exc_info=True)

    def get_vector(self, vector_id):
        try:
            return self.connection.get(vector_id)
        except Exception as e:
            self.logger.error(f"Failed to retrieve vector {vector_id}", exc_info=True)
            return None

    def update_vector(self, vector_id, new_vector_data):
        try:
            self.connection.update(vector_id, new_vector_data)
        except Exception as e:
            self.logger.error(f"Failed to update vector {vector_id}", exc_info=True)

    def delete_vector(self, vector_id):
        try:
            self.connection.delete(vector_id)
        except Exception as e:
            self.logger.error(f"Failed to delete vector {vector_id}", exc_info=True)
