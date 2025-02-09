"use client"
import { useState, useEffect } from "react";
import Image from "next/image";
import { denoise } from "@/actions";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FrameTimeline } from "@/components/frames";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge";
import { CardStack } from "@/components/ui/card-stack"
import { ImagePlayer } from "@/components/video-player"
import { Loader2 } from "lucide-react"
import { Separator } from "@/components/ui/separator"
import { FileUpload } from "@/components/ui/file-upload";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { toast } from "sonner"
import { useEditor, getAllStepsWithFrames } from "./hooks/useEditor";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"



export function Editor() {
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState("FLUX");
  const url = model === "FLUX" ? "/ws-flux" : "/ws";
  const [needPrepareLatentUpdate, setNeedPrepareLatentUpdate] = useState(false);
  const { 
    isConnected,
    output,
    prepareLatents,
    onSample,
  } = useEditor(url);
  const [isLoading, setIsLoading] = useState(false);


  const runSample = () => {
    try {
      setIsLoading(true);
      if (prompt === "") {
        throw new Error("Prompt is empty");
      }

      if (needPrepareLatentUpdate) {
        prepareLatents({ prompt });
        setNeedPrepareLatentUpdate(false);
      }

      onSample()

      //throw success toast
      toast.success("Creating sample. This may take awhile")
    } catch (e) {
      // throw toast
      toast.error(e?.message || "Something went wrong");
    } finally {
      setIsLoading(false);
    }
  }


  useEffect(() => {
    setNeedPrepareLatentUpdate(true);
  },[prompt])


  return (
    <div className="container mx-auto">
      <div>
        Selected Model: {model}
        <Button onClick={() => setModel("FLUX")}>
          Flux
        </Button>
        <Button onClick={() => setModel("SD")}>
          SD 2.1
        </Button>
      </div>

      <div className="flex flex-col gap-2 my-2 max-w-[540px] mx-auto">
        <p className="text-sm">
          Wait until the status shows "Connected", before typing in your prompt and clicking on sample.
        </p>
        <div>
          {isConnected ? <Badge className="bg-green-500">Connected</Badge> : <Badge className="bg-gray-500">Not Connected</Badge>}
        </div>
        <Input value={prompt} placeholder="A cat on the road" onChange={(e) => setPrompt(e.target.value)} />
        <Button 
          disabled={isLoading}
          onClick={() => {
          if (prompt === "") {
            return;
          }

          if (needPrepareLatentUpdate) {
            prepareLatents({ prompt });
            setNeedPrepareLatentUpdate(false);
          }
          onSample()
        }}>
          {isLoading ? <Loader2 className="animate-spin" /> : "Sample"}
        </Button>

        {output && output.images.map((img, i) => (
          <img className="max-w-[254px]" key={i} src={"data:image/png;base64," + img} />
        ))}
      </div>



      {output?.attn_maps && (
        <>
          {model === "SD" ? (
            <p className="mt-8">
              Here, we see each layer of the model's attention map. The latent runs from left to right, and the attention map for each word in the prompt is shown in the first column. The table numbers (4096, 1024, 256,... etc) represent the image area the attention is focusing upon. This is each CrossAttention from the Spatial Transformer of the UNet model. <a href="https://scholar.harvard.edu/binxuw/classes/machine-learning-scratch/materials/stable-diffusion-scratch" target="_blank">See more about Stable Diffusion UNets here.</a>
            </p>
          ) : null}
          <div className="grid grid-cols-[repeat(17,minmax(50px,1fr))] gap-4 text-center">
            {/* Row with headers (keys) */}
            <div className="contents">
              <div className="font-bold text-lg border-b pb-2">Prompt words</div>
              {Object.keys(output?.attn_maps).map((key) => (
                <div key={key} className="font-bold text-lg border-b pb-2">{key.split("_")[1]}</div>
              ))}
            </div>

            {/* Row with images under each header */}
            <div className="contents">
              <div>
                <div className="flex flex-col items-center space-y-2">
                  {prompt.split(" ").map((word, i) => (
                    <div className="h-[75px]" key={i}>{word}</div>
                  ))}
                </div>
              </div>
              {Object.keys(output?.attn_maps).map((key) => (
                <div key={key} className="flex flex-col items-center space-y-2">
                  {output?.attn_maps[key]?.map((img, i) => (
                    <img key={i} src={`data:image/png;base64,${img}`} className="w-32 h-auto" />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </>
      )}

    </div>
  );
}
