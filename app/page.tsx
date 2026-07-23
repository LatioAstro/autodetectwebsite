import LightCurve from "../components/LightCurve";
import SourceSelector from "../components/SourceSelector";
import type { LightCurveData } from "../components/LightCurve";
import fs from "node:fs/promises";
import path from "node:path";

type HomePageProps = {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
};

function getSingleSearchParam(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

async function listAvailableLightCurveFiles(): Promise<string[]> {
  const directory = path.join(process.cwd(), "data", "src_data");
  const entries = await fs.readdir(directory, { withFileTypes: true });

  return entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));
}

async function readLightCurveFromDataDir(fileName: string): Promise<LightCurveData> {
  const filePath = path.join(process.cwd(), "data", "src_data", fileName);
  const raw = await fs.readFile(filePath, "utf8");

  // Data files may contain NaN, which is not valid JSON. Convert those entries to null.
  const sanitized = raw.replace(/:\s*NaN/g, ": null");
  return JSON.parse(sanitized) as LightCurveData;
}

export default async function Home({ searchParams }: HomePageProps) {
  const files = await listAvailableLightCurveFiles();

  if (files.length === 0) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-zinc-50 px-6 text-zinc-900">
        <p>No light curve files were found in data/src_data.</p>
      </div>
    );
  }

  const requestedFile = getSingleSearchParam((await searchParams).source);
  const selectedFile = requestedFile && files.includes(requestedFile) ? requestedFile : files[0];
  const lightCurve = await readLightCurveFromDataDir(selectedFile);
  const selectedLabel = selectedFile.replace(/\.json$/i, "");

  return (
    <div className="flex flex-col flex-1 items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex flex-1 w-full max-w-5xl flex-col items-center justify-between py-20 px-8 bg-white dark:bg-black sm:items-start">
        <div className="flex w-full flex-col items-center gap-6 text-center sm:items-start sm:text-left">
          <h1 className="text-3xl font-semibold tracking-tight text-black dark:text-zinc-50">
            {selectedLabel}: Light Curve Viewer
          </h1>

          <SourceSelector files={files} selectedFile={selectedFile} />

          <LightCurve
            data={lightCurve}
            title={`${selectedLabel}: Weekly Photon Flux Light Curve`}
            className="w-full"
          />
        </div>
      </main>
    </div>
  );
}
