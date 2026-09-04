/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        border: "hsl(217 33% 17%)",
        input: "hsl(217 33% 17%)",
        ring: "hsl(224 76% 48%)",
        background: "hsl(222 47% 6%)",
        foreground: "hsl(210 40% 96%)",
        primary: { DEFAULT: "hsl(217 91% 60%)", foreground: "hsl(222 47% 11%)" },
        secondary: { DEFAULT: "hsl(217 33% 14%)", foreground: "hsl(210 40% 96%)" },
        muted: { DEFAULT: "hsl(217 33% 14%)", foreground: "hsl(215 20% 65%)" },
        accent: { DEFAULT: "hsl(217 33% 17%)", foreground: "hsl(210 40% 96%)" },
        destructive: { DEFAULT: "hsl(0 63% 31%)", foreground: "hsl(210 40% 98%)" },
        card: { DEFAULT: "hsl(222 47% 9%)", foreground: "hsl(210 40% 96%)" },
      },
      borderRadius: { lg: "0.5rem", md: "0.375rem", sm: "0.25rem" },
    },
  },
  plugins: [],
};
