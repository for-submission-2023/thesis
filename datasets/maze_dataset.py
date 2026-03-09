import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import deque
import random
from scipy.ndimage import zoom


class BinaryMazeDataset(Dataset):
    def __init__(self, transform=None, shape=224, grid_size=14, length=10000, seed=None):
        """
        Args:
            transform: Albumentations-style transform (optional)
            shape: Output image size (default 224)
            grid_size: Maze grid complexity (cells). Higher = more complex maze
            length: Number of samples in dataset
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.transform = transform
        self.shape = shape
        self.grid_size = grid_size
        self.len = length

    def _generate_maze_with_solution(self):
        """Generate maze using recursive division and solve it."""
        size = self.grid_size * 2 + 1
        maze = np.ones((size, size), dtype=np.uint8)
        
        self._recursive_division(maze, 1, 1, size - 2, size - 2)
        
        # Ensure start and end are open (harmless even if already open)
        maze[1, 1] = 0
        maze[size - 2, size - 2] = 0
        
        solution = self._solve_maze_bfs(maze)
        return maze, solution
    
    def _recursive_division(self, maze, x, y, width, height):
        """Recursive division algorithm."""
        if width < 2 or height < 2:
            return
        
        for i in range(y, y + height + 1):
            for j in range(x, x + width + 1):
                if maze[i, j] == 1:
                    maze[i, j] = 0
        
        if width <= 2 and height <= 2:
            return
        
        if width > height:
            if width >= 3:
                wall_x = random.randrange(x + 1, x + width, 2)
                gap_y = random.randrange(y, y + height + 1, 2)
                
                for i in range(y, y + height + 1):
                    if i != gap_y:
                        maze[i, wall_x] = 1
                
                self._recursive_division(maze, x, y, wall_x - x - 1, height)
                self._recursive_division(maze, wall_x + 1, y, x + width - wall_x - 1, height)
        else:
            if height >= 3:
                wall_y = random.randrange(y + 1, y + height, 2)
                gap_x = random.randrange(x, x + width + 1, 2)
                
                for j in range(x, x + width + 1):
                    if j != gap_x:
                        maze[wall_y, j] = 1
                
                self._recursive_division(maze, x, y, width, wall_y - y - 1)
                self._recursive_division(maze, x, wall_y + 1, width, y + height - wall_y - 1)
    
    def _solve_maze_bfs(self, maze):
        """BFS with parent dict to avoid O(n²) memory from path copying."""
        size = maze.shape[0]
        start = (1, 1)
        end = (size - 2, size - 2)
        
        if maze[start] == 1 or maze[end] == 1:
            return np.zeros_like(maze)
        
        queue = deque([start])
        visited = {start: None}  # node -> parent mapping
        
        while queue:
            node = queue.popleft()
            
            if node == end:
                # Reconstruct path from end to start using parent pointers
                solution = np.zeros_like(maze)
                current = node
                while current is not None:
                    solution[current] = 1
                    current = visited[current]
                return solution
            
            y, x = node
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < size and 0 <= nx < size and 
                    maze[ny, nx] == 0 and (ny, nx) not in visited):
                    visited[(ny, nx)] = node
                    queue.append((ny, nx))
        
        return np.zeros_like(maze)
    
    def _grid_to_image(self, grid):
        """Upscale binary grid to target shape."""
        zoom_factor = self.shape / grid.shape[0]
        img = zoom(grid, zoom_factor, order=0)
        return (img > 0.5).astype(np.float32)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        maze_grid, solution_grid = self._generate_maze_with_solution()
        
        # Zoom both grids - they have same shape so zoom factor is identical
        # This ensures alignment between maze and solution
        maze_img = self._grid_to_image(maze_grid)
        solution_img = self._grid_to_image(solution_grid)
        
        # RGB image: walls as white, paths as black
        image = torch.zeros((3, self.shape, self.shape))
        # image[0] = torch.from_numpy(maze_img)
        # image[1] = torch.from_numpy(maze_img)
        # image[2] = torch.from_numpy(maze_img)
        
        image[0] = torch.from_numpy(1.0 - maze_img)
        image[1] = torch.from_numpy(1.0 - maze_img)
        image[2] = torch.from_numpy(1.0 - maze_img)


        # Binary mask: path = 1, background = 0
        mask = torch.from_numpy(solution_img).long()
        
        if self.transform is not None:
            image = image.permute(1, 2, 0).numpy()
            mask = mask.numpy()
            transformed = self.transform(image=image, mask=mask)
            image = transforms.ToTensor()(transformed["image"]).float()
            mask = torch.from_numpy(transformed["mask"]).long()
        
        return image, mask
    
# import albumentations as A
# import matplotlib
# matplotlib.use('TkAgg')  # Necessary to run matplotlib
# import matplotlib.pyplot as plt

# IMG_SIZE = 224

# # Transforms
# train_transform = A.Compose([
#     A.Resize(IMG_SIZE, IMG_SIZE),
#     # A.HorizontalFlip(p=AUG_PROB),
#     # A.VerticalFlip(p=AUG_PROB),
#     # A.Rotate(limit=ROTATION_LIMIT, p=AUG_PROB),
# ])

# valid_transform = A.Compose([
#     A.Resize(IMG_SIZE, IMG_SIZE),
# ])

# # Create datasets



# train_dataset = BinaryMazeDataset(transform=valid_transform,
#                                   shape=224,
#                                   grid_size=14,
#                                   length=500,
#                                   seed=42)


# import random

# for i in range(10):
#     x = random.randint(0, len(train_dataset) - 1)
#     a, b = train_dataset[x]
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(a.permute(1, 2, 0))
#     plt.subplot(1, 2, 2)
#     plt.imshow(b)
#     plt.show()