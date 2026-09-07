export default function AlertListPage() {
  return (
    <main className="mx-auto w-full max-w-5xl flex-1 bg-white px-8 py-16 dark:bg-black">
      <div className="max-w-2xl">
        <p className="mb-3 text-sm font-medium uppercase tracking-wide text-cyan-700 dark:text-cyan-400">
          Notifications
        </p>
        <h1 className="text-4xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
          Alert list
        </h1>
        <p className="mt-5 text-lg leading-8 text-zinc-600 dark:text-zinc-300">
          Potential flare alerts will appear here as the monitoring workflow is connected.
        </p>
        <h1 className="text-4xl font-semibold tracking-tight text-zinc-950 dark:text-zinc-50">
          Sign-Ups
        </h1>
        <p className="mt-5 text-lg leading-8 text-zinc-600 dark:text-zinc-300">
          Users can sign up for weekly alerts for potential sources through contacting Garrett Latiolais via Slack
          or through email, or through the Google Forms link <a href="https://docs.google.com/forms/d/e/1FAIpQLSf4kC6VIFV9vtOTjVgkYln4v13tNGlStMYN9-fYPiYIX_iciA/viewform?usp=publish-editor" className="text-cyan-700 dark:text-cyan-400 underline">here </a>.
        </p>
      </div>
    </main>
  );
}
