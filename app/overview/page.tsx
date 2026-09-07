export default function OverviewPage() {
  return (
    <main className="mx-auto w-full max-w-5xl flex-1 bg-white px-8 py-16 dark:bg-black">
      <div className="max-w-2xl">
        <p className="mb-3 text-sm font-medium uppercase tracking-wide text-cyan-700 dark:text-cyan-400">
          COSI flare monitoring
        </p>
        <h1 className="text-4xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
          Overview of this tool
        </h1>
        <p className="mt-5 text-lg leading-8 text-zinc-600 dark:text-zinc-300">
          The primary goal of this tool is to be a visual companion to the automated alert system developed for monitoring potential blazar flares within COSI's energy band.
        </p>
        <h1 className="text-4xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
          Getting Started
        </h1>
        <p className="mt-5 text-lg leading-8 text-zinc-600 dark:text-zinc-300">
          Before using this tool, it is important to understand that it is 
          designed to complement the automated alert system, not be a substitute for it. 
          The best way to receive up-to-date information about potential flaring sources is through signing up for alerts directly. 
          This website is still in development, and is subject to change. 
          You can read about new updates on 
          the <a href="https://github.com/LatioAstro/autodetectwebsite" className="text-cyan-700 dark:text-cyan-400 underline">Github Repo</a> here.
        </p>
      </div>
    </main>
  );
}
