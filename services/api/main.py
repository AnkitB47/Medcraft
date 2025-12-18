from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import httpx
import os
import redis
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
import time

app = FastAPI(title="MedCraft API Gateway")

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://localhost:8001")
NANOVLM_SERVICE_URL = os.getenv("NANOVLM_SERVICE_URL", "http://localhost:8003")
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8002")

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Redis for Streams and Rate Limiting
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Auth
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # In production, verify against DB
    if form_data.username != "admin" or form_data.password != "admin":
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return {"username": username}

# Rate Limiting Middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    # Simple rate limit: 100 requests per minute
    key = f"rate_limit:{client_ip}"
    current = r.get(key)
    
    if current and int(current) > 100:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, 60)
    pipe.execute()
    
    response = await call_next(request)
    return response

from fastapi.responses import JSONResponse

@app.get("/healthz")
async def healthz():
    return {"status": "healthy"}

# Routing to Vision Service
@app.post("/api/vision/{module}")
async def vision_proxy(module: str, file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    async with httpx.AsyncClient() as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        response = await client.post(f"{VISION_SERVICE_URL}/vision/{module}", files=files)
        return response.json()

# Routing to NanoVLM Service
@app.post("/api/reasoning/qa")
async def reasoning_proxy(prompt: str, file: Optional[UploadFile] = File(None), user: dict = Depends(get_current_user)):
    async with httpx.AsyncClient() as client:
        data = {"prompt": prompt}
        files = None
        if file:
            files = {"file": (file.filename, await file.read(), file.content_type)}
        response = await client.post(f"{NANOVLM_SERVICE_URL}/reasoning/qa", data=data, files=files)
        return response.json()

# Async Job Submission (Redis Streams)
@app.post("/api/jobs/submit")
async def submit_job(job_type: str, payload: dict, user: dict = Depends(get_current_user)):
    job_id = r.xadd("medcraft_jobs", {
        "type": job_type,
        "payload": str(payload),
        "user": user["username"],
        "timestamp": str(datetime.now())
    })
    return {"job_id": job_id, "status": "submitted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
