import { writable } from 'svelte/store';

// Define API base URL
const API_BASE_URL = 'http://localhost:5555';

interface StreamState {
  isVisible: boolean;
  url: string | null;
  streamKey: string | null;
}

const initialState: StreamState = {
  isVisible: false,
  url: API_BASE_URL,
  streamKey: null,
};

export const streamStore = writable<StreamState>(initialState);

// Function to fetch available streams from the server
export async function fetchAvailableStreams(): Promise<string[]> {
  try {
    console.log('Fetching available streams...');
    const response = await fetch(`${API_BASE_URL}/streams`);
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Available streams:', data.streams);
    return data.streams;
  } catch (error) {
    console.error('Failed to fetch available streams:', error);
    throw error;
  }
}

// Function to start a stream
export function startStream(streamKey: string): void {
  streamStore.set({
    isVisible: true,
    url: API_BASE_URL,
    streamKey: streamKey
  });
}

// Function to stop a stream
export function stopStream(): void {
  streamStore.set({
    ...initialState
  });
}
