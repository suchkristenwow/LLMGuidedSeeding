import open3d as o3d

def view(filename: str) -> None:
    """Read a .pcd file and draw it with open3d"""
    pcd = o3d.io.read_point_cloud(filename)
    o3d.visualization.draw_geometries([pcd])


if __name__== "__main__":
    view('projected_pcd.pcd')