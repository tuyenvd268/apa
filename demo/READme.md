### Pronunciation scoring
1. Build docker images:
    ```
    docker build -t prep/pykaldi-gpu:latest .
    ```

2. Run:
    ```
    sudo nvidia-docker run -it --gpus '"device=0,1"' -p 6868:6868 \
        -v /data/codes/prep_ps_pykaldi:/data/codes/prep_ps_pykaldi \
        --name scoring prep/pykaldi-gpu:latest
    ```

    ```
    sudo nvidia-docker run -it --gpus '"device=0,1"' -p 8888:8888 \
        -v /data/codes/prep_ps_pykaldi:/data/codes/prep_ps_pykaldi \
        --name extract_feature prep/pykaldi-gpu:latest
    ```
3. Run (demo directory):

    ```
    bash run_scoring_api.sh
    ```

    ```
    bash run_force_align_api.sh
    ```

    ```
    python main.py
    ```

4. Output format: 

    ```
        """ 
            {
                "text": "...",
                "arpabet": "...",
                "score": "...",
                "words": [
                    {
                        "text": "...",
                        "arpabet": "...",
                        "score": "...",
                        "phones": [
                            {
                                "arpabet": "...",
                                "ipa": "...",
                                "score": "..."
                            },
                            {
                                "arpabet": "...",
                                "ipa": "...",
                                "score": "..."
                            }
                        ]
                    },
                ]
            }
        """
    ```