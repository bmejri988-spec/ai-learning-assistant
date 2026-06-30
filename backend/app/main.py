from fastapi import FastAPI

app = FastAPI(title="AI Learning Assistant API")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AI Learning Assistant API"}
