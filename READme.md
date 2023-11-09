### Pronunciation scoring
build docker images:
```
docker build -t prep/pykaldi:latest .
```

run:
```
sudo nvidia-docker run -it --gpus '"device=0,1"' -p 6868:6868 \
    -v /data/codes/prep_ps_pykaldi:/data/codes/prep_ps_pykaldi \
    prep/pykaldi:latest
```

```
sudo nvidia-docker run -it --gpus '"device=0,1"' \
    -v /data/codes/prep_ps_pykaldi:/data/codes/prep_ps_pykaldi \
    pykaldi/pykaldi:latest
```