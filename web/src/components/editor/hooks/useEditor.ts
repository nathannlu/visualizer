import { useState, useEffect } from "react";
import { WebSocketManager } from "@/lib/websocket";

function toTwoDigits(num) {
  return String(num).padStart(2, "0");
}

export function getAllStepsWithFrames(dict, frame) {
  const framesForStep = []
  const numSteps = Object.keys(dict).length;
  for (let i = 0; i < numSteps; i++) {
    const f = dict[i][frame];
    framesForStep.push(f);
  }

  return framesForStep;
}


/**
 * Manages sending and receiving data from websocket
 */
export const useEditor = (url) => {
  const [output, setOutput] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  const [step, setStep] = useState(0);

  const wsOpts = {
    onMessage: (data) => {
      const parsed = JSON.parse(data);
      if (parsed.type === "on_sample") {
        console.log("set output")
        const { data } = parsed;
        setOutput(data);
        if (step <= 25) {
          onSample()
        }
      }
    },
    onOpen: () => {
      setIsConnected(true) 
      console.log("WebSocket connected")
    },
    onClose: () => {
      setIsConnected(false)
      console.log("WebSocket disconnected")
    },
    onError: (error) => console.error("WebSocket error:", error),
  }

  // Set up websocket
  const websocketUrl = "ws://localhost:8000" + url
  const [wsManager] = useState(new WebSocketManager(websocketUrl, wsOpts));
  useEffect(() => {
    wsManager.connect();

    return () => {
      wsManager.disconnect();
    };
  }, [url]);


  const prepareLatents = (data) => {
    // data needs
    // - prompt
    // - negative_prompt
    // - steps

    wsManager.sendMessage({ type: "prepare_latents", data })
  }


  // The argument steps denotes which steps to export the video
  // with.
  const onSample = (data) => {
    wsManager.sendMessage({type: "on_sample", data})
    setStep(step + 1);
  }


  return {
    prepareLatents,
    onSample,
    output,
    isConnected,

  }
}
