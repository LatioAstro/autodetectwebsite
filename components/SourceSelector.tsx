"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useTransition } from "react";

type SourceSelectorProps = {
  files: string[];
  selectedFile: string;
};

export default function SourceSelector({ files, selectedFile }: SourceSelectorProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [isPending, startTransition] = useTransition();

  return (
    <div className="flex w-full flex-wrap items-end gap-3">
      <label htmlFor="source" className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
        Data source
      </label>
      <select
        id="source"
        name="source"
        value={selectedFile}
        disabled={isPending}
        onChange={(event) => {
          const nextSource = event.target.value;
          const nextParams = new URLSearchParams(searchParams.toString());
          nextParams.set("source", nextSource);

          startTransition(() => {
            router.push(`${pathname}?${nextParams.toString()}`, { scroll: false });
          });
        }}
        className="min-w-[18rem] rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 disabled:cursor-not-allowed disabled:opacity-70 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
      >
        {files.map((file) => (
          <option key={file} value={file}>
            {file.replace(/\.json$/i, "")}
          </option>
        ))}
      </select>
      <span
        aria-live="polite"
        className="min-w-20 text-xs font-medium text-zinc-500 dark:text-zinc-400"
      >
        {isPending ? "Loading..." : ""}
      </span>
    </div>
  );
}
