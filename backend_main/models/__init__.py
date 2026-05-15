from backend_main.config import Base, engine
from backend_main.models.user import User, LoginHistory
from backend_main.models.media import MediaAsset
from backend_main.models.project import Project, ProjectMediaAsset
from backend_main.models.reel import ReelJob

Base.metadata.create_all(bind=engine)
