"use client"

import { useState } from "react"
import Image from "next/image"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"

interface Frame {
  id: string
  src: string
}

interface FrameTimelineProps {
  frames: Frame[]
  onFrameSelect: (frameId: string) => void
}

export function FrameTimeline({ frames, onFrameSelect }: FrameTimelineProps) {
  const [selectedFrames, setSelectedFrames] = useState<Set<string>>(new Set())

  const toggleFrameSelection = (frameId: string) => {
    setSelectedFrames((prev) => {
      const newSelection = new Set(prev)
      if (newSelection.has(frameId)) {
        newSelection.delete(frameId)
      } else {
        newSelection.add(frameId)
      }
      return newSelection
    })
    onFrameSelect(frameId)
  }

  return (
    <div className="w-full bg-gray-900 p-4">
      <ScrollArea className="w-full whitespace-nowrap rounded-md border">
        <div className="flex space-x-2 p-4">
          {frames.map((frame) => (
            <div
              key={frame.id}
              className={`relative flex-shrink-0 cursor-pointer rounded-md border-2 ${
                selectedFrames.has(frame.id) ? "border-blue-500" : "border-transparent"
              }`}
              onClick={() => toggleFrameSelection(frame.id)}
            >
              <Image
                src={frame.src || "/placeholder.svg"}
                alt={`Frame ${frame.id}`}
                width={120}
                height={68}
                className="rounded-md object-cover"
              />
              <div className="absolute bottom-1 right-1 rounded-full bg-black bg-opacity-50 px-2 py-1 text-xs text-white">
                {frame.id}
              </div>
            </div>
          ))}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  )
}


