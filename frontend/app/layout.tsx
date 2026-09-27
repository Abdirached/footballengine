import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FootballEngine — Tactical Analysis",
  description:
    "Positional-play engine for 20Hz tracking data: triangles, pitch control, pressing, move evaluation.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-neutral-950 text-neutral-100 antialiased">
        {children}
      </body>
    </html>
  );
}
