"use client"
import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ChevronLeft, ChevronRight } from "lucide-react"

type Card = {
  id: number
  name: string
  designation: string
  content: React.ReactNode
}

export const CardStack = ({
  items,
  offset,
  scaleFactor,
}: {
  items: Card[]
  offset?: number
  scaleFactor?: number
}) => {
  const CARD_OFFSET = offset || 10
  const SCALE_FACTOR = scaleFactor || 0.06
  const VISIBLE_CARDS = 5

  const [currentIndex, setCurrentIndex] = useState(0)
  const [isPopupOpen, setIsPopupOpen] = useState(false)

  const visibleCards = items.slice(currentIndex, currentIndex + VISIBLE_CARDS)

  const moveCardToFront = (index: number) => {
    setCurrentIndex(index)
    setIsPopupOpen(false)
  }

  const nextCard = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1 >= items.length ? 0 : prevIndex + 1))
  }

  const prevCard = () => {
    setCurrentIndex((prevIndex) => (prevIndex - 1 < 0 ? items.length - 1 : prevIndex - 1))
  }

  return (
    <div className="relative h-60 w-60 md:h-60 md:w-96">
      {visibleCards.map((card, index) => (
        <motion.div
          key={card.id}
          className="absolute dark:bg-black bg-white h-60 w-60 md:h-60 md:w-96 rounded-3xl p-4 shadow-xl border border-neutral-200 dark:border-white/[0.1] shadow-black/[0.1] dark:shadow-white/[0.05] flex flex-col justify-between"
          style={{
            transformOrigin: "top center",
          }}
          animate={{
            top: index * -CARD_OFFSET,
            scale: 1 - index * SCALE_FACTOR,
            zIndex: VISIBLE_CARDS - index,
          }}
        >
          <div className="font-normal text-neutral-700 dark:text-neutral-200">{card.content}</div>
          <div>
            <p className="text-neutral-500 font-medium dark:text-white">{card.name}</p>
            <p className="text-neutral-400 font-normal dark:text-neutral-200">{card.designation}</p>
          </div>
        </motion.div>
      ))}
      <div className="absolute bottom-[-60px] left-1/2 transform -translate-x-1/2 flex items-center space-x-4">
        <button className="bg-blue-500 text-white p-2 rounded-full" onClick={prevCard} aria-label="Previous card">
          <ChevronLeft size={24} />
        </button>
        <button className="bg-blue-500 text-white px-4 py-2 rounded-md" onClick={() => setIsPopupOpen(true)}>
          Select Card
        </button>
        <button className="bg-blue-500 text-white p-2 rounded-full" onClick={nextCard} aria-label="Next card">
          <ChevronRight size={24} />
        </button>
      </div>
      <AnimatePresence>
        {isPopupOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="absolute top-[-60px] left-1/2 transform -translate-x-1/2 bg-white dark:bg-gray-800 p-4 rounded-md shadow-lg z-50"
          >
            <div className="flex flex-wrap gap-2 max-w-[300px]">
              {items.map((_, index) => (
                <button
                  key={index}
                  className="bg-blue-500 text-white px-3 py-1 rounded-md"
                  onClick={() => moveCardToFront(index)}
                >
                  {index + 1}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}


