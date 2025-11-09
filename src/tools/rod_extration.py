#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import networkx as nx

# skimage's 3D skeletonize moved/removed in some versions. Try to import
# the explicit 3D function and fall back to a per-slice 2D skeletonize if
# unavailable (less accurate for true 3D skeletons).
try:
    from skimage.morphology import skeletonize_3d
    _HAS_SKELETONIZE_3D = True
except ImportError:
    try:
        # In newer scikit-image versions, skeletonize can work on 3D
        from skimage.morphology import skeletonize
        # Test if it supports 3D
        _test = skeletonize(np.ones((3,3,3), dtype=bool))
        skeletonize_3d = skeletonize
        _HAS_SKELETONIZE_3D = True
    except Exception:
        from skimage.morphology import skeletonize as _skeletonize_2d
        _HAS_SKELETONIZE_3D = False
# --- Mesh loading -------------------------------------------------------------
def load_mesh_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".stl", ".obj", ".ply", ".glb", ".gltf"]:
        import trimesh
        mesh = trimesh.load(path, force='mesh')
        
        # Handle case where trimesh.load returns a Scene or other type
        if isinstance(mesh, trimesh.Scene):
            # Extract the first geometry from the scene
            if len(mesh.geometry) == 0:
                raise ValueError(f"Scene loaded from {path} contains no geometry.")
            # Get the first mesh from the scene
            mesh = list(mesh.geometry.values())[0]
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Failed to load a valid mesh from {path}. Got type: {type(mesh)}")
        
        if mesh.is_empty:
            raise ValueError("Loaded mesh is empty.")
        try:
            if not mesh.is_watertight:
                # fill_holes() modifies in place and returns None in some versions
                # or returns a boolean success flag in others. Don't reassign.
                mesh.fill_holes()
        except Exception:
            pass
        return mesh
    elif ext in [".step", ".stp"]:
        try:
            from OCC.Extend.DataExchange import read_step_file
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            # Some pythonocc builds expose different names for TopoDS
            # helpers. To maximize compatibility, don't rely on a
            # specific caster function; `exp.Current()` already returns
            # a TopoDS_Shape that can be passed to BRep_Tool.Triangulation.
            # Avoid importing `topods_Face` which may not exist in all
            # pythonocc distributions.
            from OCC.Core.TopLoc import TopLoc_Location
            from OCC.Core.BRep import BRep_Tool
        except Exception as e:
            raise RuntimeError(
                "STEP requires pythonocc-core (or convert to STL/OBJ/PLY)."
            ) from e
        import trimesh

        shape = read_step_file(path)
        BRepMesh_IncrementalMesh(shape, 0.5).Perform()

        vertices, faces = [], []

        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            # exp.Current() returns the current TopoDS_Shape (a face
            # in this explorer). Use it directly; avoids relying on
            # version-specific caster names.
            face = exp.Current()
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation(face, loc)
            if tri:
                # Different pythonocc / OCC versions expose different
                # Poly_Triangulation APIs. Try the common Nodes()/Triangles()
                # pattern first, else fall back to NbNodes/Node and
                # NbTriangles/Triangle. Build small accessors so the
                # downstream code is uniform.
                T = loc.Transformation()
                
                # Offset for vertices in this face
                v_offset = len(vertices)
                
                try:
                    nodes = tri.Nodes()
                    tris = tri.Triangles()
                    n_nodes = nodes.Size()
                    n_tris = tris.Size()
                    get_node = lambda i: nodes.Value(i)
                    get_tri = lambda i: tris.Value(i).Get()
                except Exception:
                    # fallback variants
                    if hasattr(tri, 'NbNodes') and hasattr(tri, 'Node'):
                        n_nodes = tri.NbNodes()
                        get_node = lambda i: tri.Node(i)
                    elif hasattr(tri, 'Nodes'):
                        nodes = tri.Nodes()
                        n_nodes = nodes.Size()
                        get_node = lambda i: nodes.Value(i)
                    else:
                        n_nodes = 0
                        get_node = lambda i: None

                    if hasattr(tri, 'NbTriangles') and hasattr(tri, 'Triangle'):
                        n_tris = tri.NbTriangles()
                        get_tri = lambda i: tri.Triangle(i).Get()
                    elif hasattr(tri, 'Triangles'):
                        tris = tri.Triangles()
                        n_tris = tris.Size()
                        get_tri = lambda i: tris.Value(i).Get()
                    else:
                        n_tris = 0
                        get_tri = lambda i: (0,0,0)

                # Add all vertices from this face
                for i in range(1, int(n_nodes)+1):
                    p = get_node(i)
                    # most APIs return a gp_Pnt so Transformed(T) works;
                    # guard in case the point object differs.
                    try:
                        p_t = p.Transformed(T)
                    except Exception:
                        p_t = p
                    vertices.append([p_t.X(), p_t.Y(), p_t.Z()])

                # Add triangles with adjusted indices
                for i in range(1, int(n_tris)+1):
                    a, b, c = get_tri(i)
                    # Indices in OCC are 1-based, adjust to 0-based and add offset
                    faces.append([v_offset + a - 1, v_offset + b - 1, v_offset + c - 1])
            exp.Next()

        if not vertices or not faces:
            raise ValueError("Failed to tessellate STEP; try finer meshing or convert.")
        return trimesh.Trimesh(vertices=np.array(vertices, float),
                               faces=np.array(faces, int),
                               process=True)
    else:
        raise ValueError(f"Unsupported file: {ext}")

# --- Voxelization & skeletonization ------------------------------------------
def mesh_to_solid_voxels(mesh, voxel_size):
    import trimesh
    v = mesh.voxelized(pitch=voxel_size)
    v = v.fill(method='orthographic')
    # Return the voxel grid matrix, the transformation matrix, and pitch
    return v.matrix.astype(bool), v.transform, float(voxel_size)

def volume_skeleton(volume_bool):
    v = volume_bool.astype(np.uint8)
    if _HAS_SKELETONIZE_3D:
        return skeletonize_3d(v).astype(bool)
    # fallback: apply 2D skeletonize per axial slice (z-axis). This is
    # not a true 3D skeletonization but avoids ImportError on older/newer
    # scikit-image versions. Recommend installing a scikit-image that
    # exposes `skeletonize_3d` for correct 3D results.
    print("Warning: skimage.morphology.skeletonize_3d not available; falling back to per-slice 2D skeletonize", file=sys.stderr)
    out = np.zeros_like(v, dtype=bool)
    # assume first axis is z; iterate over slices
    for iz in range(v.shape[0]):
        out[iz] = _skeletonize_2d(v[iz]).astype(bool)
    return out

# --- Graph utilities ----------------------------------------------------------
NEIGH_OFFSETS = np.array([[dz,dy,dx] for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1)
                          if not (dz==0 and dy==0 and dx==0)], int)

def skeleton_to_graph(skel):
    idx = np.flatnonzero(skel.ravel())
    if idx.size == 0:
        return nx.Graph(), None, None
    shape = skel.shape
    zyx = np.column_stack(np.unravel_index(idx, shape))
    key = zyx[:,0]*(shape[1]*shape[2]) + zyx[:,1]*shape[2] + zyx[:,2]
    lut = {int(k): i for i,k in enumerate(key)}
    G = nx.Graph()
    G.add_nodes_from(range(len(key)))
    Z,Y,X = shape
    for i,(zz,yy,xx) in enumerate(zyx):
        for dz,dy,dx in NEIGH_OFFSETS:
            nz,ny,nx_ = zz+dz, yy+dy, xx+dx
            if 0<=nz<Z and 0<=ny<Y and 0<=nx_<X:
                lin2 = int(nz*(Y*X)+ny*X+nx_)
                j = lut.get(lin2)
                if j is not None:
                    G.add_edge(i,j)
    return G, zyx, idx

def largest_component(G):
    if G.number_of_nodes()==0: return G
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(comps[0]).copy()

def longest_path_bfs(G):
    if G.number_of_nodes()==0: return []
    leaves = [n for n,d in G.degree() if d==1]
    start = leaves[0] if leaves else next(iter(G.nodes))
    d1 = nx.single_source_shortest_path_length(G, start)
    u = max(d1, key=d1.get)
    d2 = nx.single_source_shortest_path_length(G, u)
    v = max(d2, key=d2.get)
    return nx.shortest_path(G, u, v)

# --- Centerline core ----------------------------------------------------------
def centerline_from_mesh(mesh, voxel_size=1.0, smooth_sigma=1.0):
    vol, transform, pitch = mesh_to_solid_voxels(mesh, voxel_size)
    if vol.sum()==0:
        raise RuntimeError("Voxelization produced empty volume; decrease --voxel.")
    skel = volume_skeleton(vol)
    G, zyx, _ = skeleton_to_graph(skel)
    Gc = largest_component(G)
    path = longest_path_bfs(Gc)
    if len(path)<2:
        raise RuntimeError("Could not find a valid centerline path; adjust --voxel.")
    # compact id -> zyx mapping
    zyx_map = {i:z for i,z in zip(Gc.nodes(), zyx[list(Gc.nodes())])}
    pts = []
    # Extract origin and pitch from transform
    origin = transform[:3, 3]  # translation component
    scale = pitch  # the voxel pitch
    
    for nid in path:
        iz, iy, ix = zyx_map[nid]
        # Voxel matrix indices [iz, iy, ix] need to map to world [x, y, z]
        # Trimesh voxel grids are stored with shape (nx, ny, nz) 
        # and indexed as matrix[i, j, k] where the first index corresponds to X axis
        # So: iz->x, iy->y, ix->z
        world = origin + np.array([iz * scale, iy * scale, ix * scale])
        pts.append(world)
    pts = np.vstack(pts)
    if smooth_sigma>0 and pts.shape[0]>=7:
        from scipy.ndimage import gaussian_filter1d
        pts = np.column_stack([gaussian_filter1d(pts[:,i], sigma=smooth_sigma, mode='nearest')
                               for i in range(3)])
    return pts

def resample_polyline(pts, n):
    if n is None or n<=0: return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if seg.size==0: return pts
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    out = np.column_stack([np.interp(t, s, pts[:,i]) for i in range(3)])
    return out

# --- Visualization (Plotly - better Wayland support) -------------------------
def visualize(mesh, centerline_pts):
    """Interactive 3D visualization using plotly (works with Wayland)"""
    import plotly.graph_objects as go
    
    # Prepare mesh data
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    
    # Create mesh trace (light gray surface)
    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightgray',
        opacity=0.5,
        name='Mesh',
        flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
        lightposition=dict(x=100, y=200, z=0)
    )
    
    # Create centerline trace (red line)
    centerline_trace = go.Scatter3d(
        x=centerline_pts[:, 0],
        y=centerline_pts[:, 1],
        z=centerline_pts[:, 2],
        mode='lines+markers',
        line=dict(color='red', width=6),
        marker=dict(size=3, color='darkred'),
        name='Centerline'
    )
    
    # Create figure
    fig = go.Figure(data=[mesh_trace, centerline_trace])
    
    # Update layout for better 3D viewing
    fig.update_layout(
        title='Centerline Extraction Preview',
        scene=dict(
            xaxis=dict(title='X', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            yaxis=dict(title='Y', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            zaxis=dict(title='Z', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            aspectmode='data'
        ),
        width=1280,
        height=800,
        showlegend=True
    )
    
    # Open in browser
    fig.show()

# --- CLI ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract (and visualize) a centerline from a tubular mesh.")
    ap.add_argument("input", help="Mesh file (.stl/.obj/.ply; .step if pythonocc-core installed)")
    ap.add_argument("--voxel", type=float, default=1.0, help="Voxel size (model units). Smaller → finer.")
    ap.add_argument("--smooth", type=float, default=1.0, help="Gaussian 1D smoothing sigma along path.")
    ap.add_argument("--resample", type=int, default=0, help="Resample centerline to N evenly spaced points.")
    ap.add_argument("--out", default=None, help="Output CSV (default: <input>_centerline.csv)")
    ap.add_argument("--npy", default=None, help="Also save .npy (default: <input>_centerline.npy)")
    ap.add_argument("--show", action="store_true", help="Open interactive 3D viewer with mesh + centerline (plotly in browser).")
    args = ap.parse_args()

    mesh = load_mesh_any(args.input)

    # Auto voxel heuristic if <=0: ~1/200 of bbox diag
    if args.voxel <= 0:
        bbox = mesh.bounds
        diag = np.linalg.norm(bbox[1]-bbox[0])
        args.voxel = max(diag/200.0, 1e-4)

    pts = centerline_from_mesh(mesh, voxel_size=args.voxel, smooth_sigma=args.smooth)
    if args.resample and args.resample>1:
        pts = resample_polyline(pts, args.resample)

    base = os.path.splitext(args.input)[0]
    csv_path = args.out or f"{base}_centerline.csv"
    npy_path = args.npy or f"{base}_centerline.npy"
    np.savetxt(csv_path, pts, delimiter=",", header="x,y,z", comments="")
    np.save(npy_path, pts)

    print(f"Centerline points: {pts.shape[0]}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved NPY: {npy_path}")

    if args.show:
        # Lazy import so headless servers can still run extraction only
        import trimesh
        visualize(mesh, pts)

if __name__ == "__main__":
    main()
