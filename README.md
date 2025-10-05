# Solyd Team Feedstock Screening – ML + Conveyor Routing


pyrolysis_screening/
│
├── model.py                # Mask R-CNN architecture builder
├── utils/
│   ├── camera_stream.py    # OpenCV camera interface
│   ├── actuator.py         # Pneumatic/servo control
│   ├── viz.py              # FPS overlay and HUD
│
├── main.py                 # Live detection & routing
├── train.py                # Fine-tuning script
├── labels.json             # Dataset label mapping
├── weights/                # Pretrained model weights
└── README.md               # Documentation
