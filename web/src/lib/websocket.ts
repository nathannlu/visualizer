type WebSocketCallbacks = {
  onOpen?: () => void;
  onMessage?: (data: string) => void; // Triggered when a message is received
  onClose?: () => void;
  onError?: (error: Event) => void;
};

export class WebSocketManager {
  private socket: WebSocket | null = null;

  constructor(private url: string, private callbacks: WebSocketCallbacks = {}) {}

  connect() {
    if (this.socket) {
      console.warn("WebSocket is already connected.");
      return;
    }

    this.socket = new WebSocket(this.url);

    this.socket.onopen = () => {
      console.log("WebSocket connected to", this.url);
      if (this.callbacks.onOpen) this.callbacks.onOpen();
    };

    this.socket.onmessage = (event) => {
      console.log("WebSocket message received:", event.data);
      if (this.callbacks.onMessage) this.callbacks.onMessage(event.data); // Call the onMessage callback
    };

    this.socket.onclose = () => {
      console.log("WebSocket disconnected");
      if (this.callbacks.onClose) this.callbacks.onClose();
      this.socket = null;
    };

    this.socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      if (this.callbacks.onError) this.callbacks.onError(error);
    };
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    } else {
      console.warn("WebSocket is not connected.");
    }
  }


  sendMessage(data: object) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {

      if (typeof data !== "object") {
        console.error("Data must be an object");
        return;
      }

      const message = JSON.stringify(data); // Convert object to JSON string
      this.socket.send(message);
    } else {
      console.warn("WebSocket is not connected.");
    }
  }

  isConnected() {
    return this.socket?.readyState === WebSocket.OPEN;
  }
}
