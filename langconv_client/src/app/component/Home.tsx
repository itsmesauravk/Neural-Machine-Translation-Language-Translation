"use client"
import React, { useState, ChangeEvent } from "react"

interface TranslationResponse {
  translated_text: string
  language: "nepali" | "german"
  english_text: string
}

const Home: React.FC = () => {
  const [input, setInput] = useState<string>("")
  const [loading, setLoading] = useState<boolean>(false)
  const [result, setResult] = useState<TranslationResponse | null>(null)

  const handleTranslate = async (lang: string) => {
    if (!input.trim()) return
    setLoading(true)
    setResult(null)

    try {
      const res = await fetch(`http://localhost:8000/convert/${lang}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      })

      if (!res.ok) throw new Error("Failed to fetch")

      const data: TranslationResponse = await res.json()

      setResult(data)
    } catch (err: unknown) {
      console.error("Translation error:", err)
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="w-full max-w-md ">
        <h1 className="text-2xl sm:text-3xl font-bold mb-2 text-blue-600 text-center">
          langConv
        </h1>
        <p className="text-gray-600 mb-6 text-center text-sm sm:text-base">
          Convert English language to German or Nepali
        </p>

        <label
          htmlFor="english-input"
          className="block text-lg font-semibold text-gray-700 mb-1"
        >
          English
        </label>
        <textarea
          id="english-input"
          rows={3}
          className="w-full text-black font-semibold p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 bg-gray-100"
          value={input}
          onChange={handleInputChange}
          placeholder="Type something in English..."
        />

        <div className="flex flex-col sm:flex-row gap-3 mt-4">
          <button
            onClick={() => handleTranslate("german")}
            disabled={loading}
            className="w-full bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 disabled:opacity-50 transition"
          >
            Convert to German
          </button>
          <button
            onClick={() => handleTranslate("nepali")}
            disabled={loading}
            className="w-full bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600 disabled:opacity-50 transition"
          >
            Convert to Nepali
          </button>
        </div>

        {loading && (
          <p className="mt-4 text-sm text-gray-500 text-center">
            Translating...
          </p>
        )}

        {result && (
          <div className="mt-6 p-4 border border-gray-200 rounded-lg bg-gray-100">
            <h3 className="text-lg font-semibold mb-2 text-gray-800">
              Translation Details:
            </h3>
            <p className="text-md mb-1 text-gray-700">
              Your Text:{" "}
              <span className="font-semibold">{result.english_text}</span>
            </p>
            <p className="text-md mb-1 text-gray-700">
              Convert to: {""}
              <span className="font-semibold ">{result.language}</span>
            </p>
            <p className="text-md text-gray-700 mt-2">
              Translation:
              <p className="font-semibold ">{result.translated_text}</p>
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default Home
