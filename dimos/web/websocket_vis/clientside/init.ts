import { io } from "npm:socket.io-client";

const socket = io();

socket.on("connect", () => {
  console.log("Connected to server");
});

socket.on("disconnect", () => {
  console.log("Disconnected from server");
});

socket.on("message", (data) => {
  console.log("Received message:", data);
});

console.log("Socket.IO client initialized");
