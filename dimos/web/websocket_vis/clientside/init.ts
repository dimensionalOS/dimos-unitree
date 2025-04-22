import { io } from "npm:socket.io-client";
import { decode, EncodedSomething } from "./decoder.ts";
import { Drawable } from "./types.ts";

// Store server state locally
let serverState = {
  status: "disconnected",
  connected_clients: 0,
  data: {},
};

const socket = io();

socket.on("connect", () => {
  console.log("Connected to server");
  serverState.status = "connected";
});

socket.on("disconnect", () => {
  console.log("Disconnected from server");
  serverState.status = "disconnected";
});

socket.on("message", (data) => {
  console.log("Received message:", data);
});

// Deep merge function for client-side state updates
function deepMerge(source: any, destination: any): any {
  for (const key in source) {
    // If both source and destination have the property and both are objects, merge them
    if (
      key in destination &&
      typeof source[key] === "object" &&
      source[key] !== null &&
      typeof destination[key] === "object" &&
      destination[key] !== null &&
      !Array.isArray(source[key]) &&
      !Array.isArray(destination[key])
    ) {
      deepMerge(source[key], destination[key]);
    } else {
      // Otherwise, just copy the value
      destination[key] = source[key];
    }
  }
  return destination;
}

type DrawConfig = { [key: string]: any };

type EncodedDrawable = EncodedSomething;
type EncodedDrawables = {
  [key: string]: EncodedDrawable;
};
type Drawables = {
  [key: string]: Drawable;
};

type Drawable = [];

function decodeDrawables(encoded: EncodedDrawables): Drawables {
  const drawbles: Drawables = {};
  for (const [key, value] of Object.entries(encoded)) {
    // @ts-ignore
    drawbles[key] = decode(value);
  }
  return drawbles;
}

function state_update(state: { [key: string]: any }) {
  console.log("Received state update:", state);
  // Use deep merge to update nested properties

  if (state.draw) {
    state.draw = decodeDrawables(state.draw);
  }

  console.log("Decoded state update:", state);
  serverState = deepMerge(state, { ...serverState });

  updateUI();
}

socket.on("state_update", state_update);
socket.on("full_state", state_update);

// Function to send data to server
function sendData(data: any) {
  socket.emit("message", data);
}

// Function to update UI based on state
function updateUI() {
  console.log("Current state:", serverState);

  // Find the element with id="json"
  const jsonElement = document.getElementById("json");

  // If the element exists, update its content with the formatted JSON
  if (jsonElement) {
    // Pretty print the JSON with 2 space indentation
    jsonElement.textContent = JSON.stringify(serverState, null, 2);
  } else {
    console.warn("Element with id='json' not found in the DOM");
  }
}

console.log("Socket.IO client initialized");

// Call updateUI once when the DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  console.log("DOM loaded, initializing UI");
  updateUI();
});
