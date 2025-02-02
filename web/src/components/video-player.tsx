"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Play, Pause, RotateCcw } from "lucide-react"

interface ImagePlayerProps {
  images: string[]
  interval?: number
}

export function ImagePlayer({ images, interval = 1000 }: ImagePlayerProps) {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    let timer: NodeJS.Timeout
    let startTime: number

    const updateProgress = () => {
      const elapsedTime = Date.now() - startTime
      setProgress((elapsedTime / interval) * 100)
    }

    if (isPlaying) {
      startTime = Date.now()
      timer = setInterval(() => {
        setCurrentIndex((prevIndex) => (prevIndex + 1) % images.length)
        setProgress(0)
        startTime = Date.now()
      }, interval)

      const progressTimer = setInterval(updateProgress, 16) // ~60fps

      return () => {
        clearInterval(timer)
        clearInterval(progressTimer)
      }
    }
  }, [isPlaying, interval, images.length])

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const reset = () => {
    setIsPlaying(false)
    setCurrentIndex(0)
    setProgress(0)
  }

  return (
    <div className="flex flex-col items-center space-y-4">
      <div className="relative w-full aspect-video bg-gray-200 rounded-lg overflow-hidden">
        <img
          src={images[currentIndex] || "/placeholder.svg"}
          alt={`Image ${currentIndex + 1}`}
          className="absolute inset-0 w-full h-full object-cover"
        />
        {/*
        <div className="absolute bottom-0 left-0 w-full h-1 bg-gray-300">
          <div
            className="h-full bg-blue-500 transition-all duration-100 ease-linear"
            style={{ width: `${progress}%` }}
          />
        </div>
        */}
      </div>
      <div className="flex items-center space-x-2">
        <Button onClick={togglePlayPause} variant="outline" size="icon">
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </Button>
        <Button onClick={reset} variant="outline" size="icon">
          <RotateCcw className="h-4 w-4" />
        </Button>
      </div>
      <Slider
        min={0}
        max={images.length - 1}
        step={1}
        value={[currentIndex]}
        onValueChange={(value) => setCurrentIndex(value[0])}
        className="w-full max-w-md"
      />
      <p className="text-sm text-gray-500">
        Image {currentIndex + 1} of {images.length}
      </p>
    </div>
  )
}


