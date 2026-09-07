"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const tabs = [
  { label: "Overview", href: "/overview" },
  { label: "Viewer", href: "/" },
  { label: "Alert List", href: "/alert-list" },
];

export default function SiteTabs() {
  const pathname = usePathname();

  return (
    <nav aria-label="Primary navigation" className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
      <div className="mx-auto flex w-full max-w-5xl items-center gap-6 px-8" role="tablist">
        {tabs.map((tab) => {
          const isActive = tab.href === "/" ? pathname === "/" : pathname.startsWith(tab.href);

          return (
            <Link
              key={tab.href}
              href={tab.href}
              aria-current={isActive ? "page" : undefined}
              className={`border-b-2 py-4 text-sm font-medium transition-colors ${
                isActive
                  ? "border-cyan-700 text-cyan-800 dark:border-cyan-400 dark:text-cyan-300"
                  : "border-transparent text-zinc-500 hover:border-zinc-300 hover:text-zinc-900 dark:text-zinc-400 dark:hover:border-zinc-600 dark:hover:text-zinc-100"
              }`}
            >
              {tab.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
