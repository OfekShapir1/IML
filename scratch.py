# Force kill any stuck Drive process
!pkill -f "drive"

# Remove the old mount folder (just local, not your files!)
!rm -rf /content/drive

# Now try to mount again
from google.colab import drive
drive.mount('/content/drive')
