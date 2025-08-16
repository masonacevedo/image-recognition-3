from duckduckgo_search import DDGS

results = DDGS().images(
    keywords="birds",
    region="wt-wt",
    safesearch="off",
    size=None,
    color="Monochrome",
    type_image=None,
    layout=None,
    license_image=None,
    max_results=100,
)

images = [result["image"] for result in results]

with open("validation_images.txt", "w+") as f:
    for image in images:
        f.write(image + "\n")

