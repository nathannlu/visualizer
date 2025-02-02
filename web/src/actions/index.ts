"use server";

export const denoise = async (step) => {
  // fetch
  const url = "https://imintifydev--custom-modal-image-fastapi-app-wrapper-dev.modal.run/sample?steps=" + step
  const data = await fetch(url)

  if (!data.ok) {
    throw new Error(`HTTP error! status: ${data.status}`);
  }

  console.log(JSON.stringify(data.json))

  return data.json()
}
