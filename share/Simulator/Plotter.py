import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon, Ellipse, Rectangle, Circle
import os
from PIL import Image
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

from typing import List, Tuple, Optional
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class ShapeDrawer_2D:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.shapes = []  # 统一存储所有图形对象
        self.ax.grid(True)

        self.error_history = {
            'iterations': [],
            'error_feas': [],
            'error_opt': []
        }

    def _add_shape(self, patch, shape_type, ** kwargs):
        """统一添加图形到存储列表"""
        shape_id = len(self.shapes)
        shape_info = {
            'patch': patch,
            'shape': shape_type,
            'id': shape_id,
            ** kwargs  # 存储其他自定义属性
        }
        self.shapes.append(shape_info)
        return shape_id

    def plot_epigraph(self, x_range=(0, 1), f_min_func=lambda x: x ** 2,
        num_points = 500, alpha = 0.3,
        facecolor = 'skyblue',
        f_color = 'b-', label = 'epigraph', title = None,
        xlim = None, ylim = None):
        """绘制x² ≤ f ≤1类型的可行域
        参数：
            x_range: x轴数据生成范围元组，默认(0,1)
            f_min_func: 下限函数，默认x²
            f_max_val: 上限值，默认1
            num_points: 采样点数量，默认500
            alpha: 填充透明度，默认0.3
            facecolor: 填充颜色，默认天蓝
            f_min_color: 下限曲线样式，默认红色虚线
            f_max_color: 上限曲线样式，默认蓝色实线
            xlim: 显示范围x轴界限，默认None（自动适配）
            ylim: 显示范围y轴界限，默认None（自动适配）
        """
        # 生成数据点
        x = np.linspace(x_range[0], x_range[1], num_points)
        f_min = f_min_func(x)
        f_max_val = f_min_func(x_range[1])
        f_max = np.full_like(x, f_max_val)

        # 填充可行域（生成PolyCollection对象）
        fill = self.ax.fill_between(x, f_min, f_max,
                                    alpha=alpha,
                                    facecolor=facecolor,
                                    label=label)

        # 绘制边界曲线
        line_min, = self.ax.plot(x, f_min, f_color)
        line_max, = self.ax.plot(x, f_max, f_color)
        # 新增x=0边界线
        x0 = 0
        y_min_x0 = f_min_func(x0)
        line_x0, = self.ax.plot([x0, x0], [y_min_x0, f_max_val], f_color)
        # 坐标范围设置（优先级：手动设置 > 自动计算）
        display_xlim = xlim if xlim is not None else x_range
        self.ax.set_xlim(display_xlim)

        if ylim is not None:
            self.ax.set_ylim(ylim)
        else:
            ymin = f_min.min() * 0.9 if len(f_min) > 0 else 0
            ymax = f_max_val * 1.1
            self.ax.set_ylim(ymin, ymax)

        # 统一注册所有图形元素（关键修改点）
        shape_id = self._add_shape(
            patch=fill,  # 将PolyCollection作为patch存储
            shape_type='epigraph',
            lines=[line_min, line_max],  # 存储边界线对象
            params={
                'x_range': x_range,
                'f_min_func': f_min_func,
                'f_max_val': f_max_val,
                'xlim': xlim,
                'ylim': ylim,
                'artist_type': 'collection'  # 标记特殊类型
            },
            alpha=alpha,
            facecolor=facecolor,
            label=label
        )

        # 标题设置
        if title:
            self.ax.set_title(title)
        else:
            default_title = f"Epigraph: {x_range[0]} ≤ x ≤ {x_range[1]} | {f_min_func.__name__} ≤ f ≤ {f_max_val}"
            self.ax.set_title(default_title)

        return shape_id
    def plot_polygon(self, A, b, xlim, ylim, alpha=0.2, edgecolor='blue',
                     facecolor='blue', label=None, title=None):
        """绘制多边形"""
        # 计算交点
        vertices = []
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                try:
                    x, y = np.linalg.solve(np.array([A[i], A[j]]), np.array([b[i], b[j]]))
                    if np.all(A @ np.array([x, y]) <= b + 1e-5):
                        vertices.append((x, y))
                except np.linalg.LinAlgError:
                    pass

        if not vertices:
            return None

        # 顶点排序
        center = np.mean(vertices, axis=0)
        angles = np.arctan2([v[1] - center[1] for v in vertices],
                            [v[0] - center[0] for v in vertices])
        sorted_vertices = np.array([v for _, v in sorted(zip(angles, vertices))])

        # 创建并添加多边形
        polygon = Polygon(sorted_vertices, closed=True, alpha=alpha,
                          edgecolor=edgecolor, facecolor=facecolor, label=label)
        patch = self.ax.add_patch(polygon)

        # 统一存储
        shape_id = self._add_shape(
            patch=patch,
            shape_type='polygon',
            vertices=sorted_vertices,
            alpha=alpha,
            edgecolor=edgecolor,
            facecolor=facecolor,
            label=label
        )

        # 设置坐标范围和标题
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        # 移除所有边框和网格
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.grid(False)


        if title:
            self.ax.set_title(title)

        return shape_id

    def plot_convex_hull(self, points, alpha=0.2, facecolor='none',
                         edgecolor='green', vertices_marker='o',
                         vertices_color='green', label=None, title=None):
        """
        绘制点集的凸包多边形

        参数:
            points: 二维点的集合，形状为(n, 2)的NumPy数组
            alpha: 填充透明度，默认0.3
            facecolor: 填充颜色，默认'lightcoral'
            edgecolor: 边界线颜色，默认'red'
            linewidth: 边界线宽度，默认2
            vertices_marker: 凸包顶点标记样式，默认's'(方块)
            vertices_color: 凸包顶点颜色，默认'red'
            label: 图例标签
            title: 标题
        """


        # 获取凸包顶点并按顺序排列
        vertices = points

        # 创建多边形对象
        polygon = Polygon(vertices, closed=True,
                          alpha=alpha,
                          facecolor=facecolor,
                          edgecolor=edgecolor,
                          label=label)

        # 添加多边形到坐标轴
        patch = self.ax.add_patch(polygon)

        # 绘制凸包顶点标记
        vertices_x = vertices[:, 0]
        vertices_y = vertices[:, 1]
        vertices_scatter = self.ax.scatter(vertices_x, vertices_y,
                                           marker=vertices_marker,
                                           color=vertices_color,
                                           edgecolors=edgecolor,
                                           s=5, zorder=3)

        # 存储多边形和顶点标记作为同一图形
        shape_id = self._add_shape(
            patch=patch,  # 多边形作为主要补丁
            shape_type='convex_hull',
            scatter=vertices_scatter,  # 存储散点标记
            params={
                'vertices': vertices,
                'points': points,
                'vertices_color': vertices_color
            },
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
            label=label
        )

        # 设置标题
        if title:
            self.ax.set_title(title)
        else:
            default_title = f"Convex Hull of {len(points)} Points"
            self.ax.set_title(default_title)

        return shape_id

    def plot_ellipse(self, Sigma, xlim, ylim, alpha=0.2, edgecolor='green',
                     facecolor='none', label=None, title=None):
        """绘制椭圆（基于协方差矩阵）"""
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        lambda1, lambda2 = eigenvalues
        v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]

        # 计算椭圆参数
        width = 2 / np.sqrt(lambda1)
        height = 2 / np.sqrt(lambda2)
        angle = np.degrees(np.arctan2(v1[1], v1[0]))

        # 创建并添加椭圆
        ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                          alpha=alpha, edgecolor=edgecolor,
                          facecolor=facecolor, label=label)
        patch = self.ax.add_patch(ellipse)

        # 统一存储
        shape_id = self._add_shape(
            patch=patch,
            shape_type='ellipse',
            sigma=Sigma,
            alpha=alpha,
            edgecolor=edgecolor,
            facecolor=facecolor,
            label=label
        )

        # 设置坐标范围和标题
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        # 移除所有边框和网格
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.grid(False)
        if title:
            self.ax.set_title(title)

        return shape_id

    def plot_circle_regions(self, theta, xlim, ylim, alpha=0.2, edgecolor='green', facecolor='none', label=None,
                            title=None):
        """绘制单位圆内部和另一圆外部的区域"""
        # 绘制单位圆内部
        circle_inner = Circle((0, 0), 1, alpha=alpha, edgecolor=edgecolor, facecolor=facecolor,label = label)
        patch_inner = self.ax.add_patch(circle_inner)
        shape_id_inner = self._add_shape(
            patch=patch_inner,
            shape_type='circle',
            center=(0, 0),
            radius=1,
            alpha=alpha,
            edgecolor=edgecolor,
            facecolor=facecolor,
            # label=label
        )
        # circle_outer = Circle((theta[0], theta[1]), theta[2], alpha=1, edgecolor=None, facecolor='white',label = label)
        # patch_outer = self.ax.add_patch(circle_outer)
        # shape_id_outer = self._add_shape(
        #     patch=patch_outer,
        #     shape_type='circle',
        #     center=(theta[0], theta[1]),
        #     radius=theta[2],
        #     alpha=1,
        #     edgecolor=None,
        #     facecolor='white',
        #     # label=label
        # )
        # 绘制另一圆的外部区域
        a, b, r = theta[0], theta[1], theta[2]
        # 创建覆盖整个绘图区域的矩形
        rect = Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                         facecolor='white', alpha=1,label = None)
        # 创建圆形路径用于剪裁
        circle_clip = Circle((a, b), r, transform=self.ax.transData,label = None)
        rect.set_clip_path(circle_clip)
        patch_rect = self.ax.add_patch(rect)
        shape_id_outer = self._add_shape(
            patch=patch_rect,
            shape_type='rectangle',
            xlim=xlim,
            ylim=ylim,
            alpha=1,
            facecolor=None,
            edgecolor=edgecolor,
            label = None,
            clip_circle={'center': (a, b), 'radius': r}
        )

        # 设置坐标范围和标题
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        # 移除所有边框和网格
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.grid(False)
        if title:
            self.ax.set_title(title)

        return shape_id_inner, shape_id_outer
    def remove_shape(self, shape_id):
        """删除指定图形"""
        for i, shape in enumerate(self.shapes):
            if shape['id'] == shape_id:
                shape['patch'].remove()
                del self.shapes[i]
                return True
        return False

    def show(self):
        """显示图形"""
        if any(shape.get('label') for shape in self.shapes):
            self.ax.legend(loc='upper right')
        plt.show()

    def save(self, filename, dpi=300, transparent=False, format='svg', show_legend=True):
        """保存图形到文件（SVG格式）"""
        if show_legend and any(shape.get('label') for shape in self.shapes):
            self.ax.legend(
                loc='upper right',
                bbox_to_anchor=(1, 1),
                frameon=False  # 无边框图例
            )

        # 确保保存为SVG格式
        if not filename.lower().endswith('.'+format):
            filename += '.'+format

        self.fig.savefig(
            filename,
            # dpi=dpi,
            transparent=transparent,
            format=format,
            bbox_inches='tight'
        )

        print(f"图形已保存到 {filename}")




class ShapeDrawer_3D:
    """
    Draw evolving 2D polygons (given as half-space constraints Ax <= b) as 3D prisms
    across an 'iteration' axis, comparing an original polygon vs. a sequence of
    approximation polygons.

    Key features:
    - Reusable helper to convert constraints -> ordered vertices
    - Robust handling of shapes (b can be (n,) or (n,1))
    - Configurable colors/alphas, axis/labels, show/save control
    - Returns (fig, ax) so you can further customize outside the class
    """

    def __init__(
        self,
        *,
        end_face_alpha: float = 0.35,
        tube_alpha: float = 0.05,
        colors: Optional[dict] = None,
        axis_off: bool = True,
        angle = (23,-106),
    ):
        self.end_face_alpha = float(end_face_alpha)
        self.tube_alpha = float(tube_alpha)
        self.axis_off = bool(axis_off)
        self.colors = {
            # original polygon visual style
            "original_tube_face": "lightblue",
            "original_tube_edge": "blue",
            "original_end_face": "darkblue",
            "original_end_edge": "navy",
            # approximations visual style
            "approx_tube_face": "lightcoral",
            "approx_tube_edge": "red",
            "approx_end_face": "darkred",
            "approx_end_edge": "maroon",
        }
        self.elev, self.azim = angle[0], angle[1]
        if colors:
            self.colors.update(colors)

    # ------------------------
    # Public utility
    # ------------------------
    @staticmethod
    def solve_inequalities_to_vertices(A: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
        """
        Convert 2D half-space constraints Ax <= b into an ordered list of polygon vertices.
        Returns:
            vertices: (m, 2) array in CCW order, or None if infeasible / empty.
        Notes:
            - Simple pairwise intersection of each line, then inside-test.
            - Works for convex polygons. (This method does not compute a convex hull
              from arbitrary scattered points; it relies on the half-space description.)
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)

        if A.ndim != 2 or A.shape[1] != 2:
            raise ValueError("A must be of shape (n, 2).")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError("b must be of shape (n,) matching A.shape[0].")

        n = A.shape[0]
        verts = []

        # Intersect each pair of constraint lines A_i x = b_i, A_j x = b_j
        for i in range(n):
            for j in range(i + 1, n):
                M = np.vstack([A[i], A[j]])  # 2x2
                rhs = np.array([b[i], b[j]])
                try:
                    xy = np.linalg.solve(M, rhs)  # (2,)
                except np.linalg.LinAlgError:
                    continue  # parallel or ill-conditioned pair

                if np.all(A @ xy <= b + 1e-5):  # inside test with small tolerance
                    verts.append(xy)

        if not verts:
            return None

        verts = np.asarray(verts)
        # Order vertices by angle around centroid
        center = verts.mean(axis=0)
        angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = verts[order]

        # Remove near-duplicates (optional stability)
        deduped = [ordered[0]]
        for k in range(1, len(ordered)):
            if np.linalg.norm(ordered[k] - deduped[-1]) > eps:
                deduped.append(ordered[k])
        if len(deduped) > 1 and np.linalg.norm(deduped[0] - deduped[-1]) <= eps:
            deduped.pop()  # drop closing duplicate

        return np.asarray(deduped).reshape(-1, 2) if len(deduped) >= 3 else None


    # ------------------------
    # Main plotting entrypoint
    # ------------------------
    def plot_polygon_evolution(
        self,
        A: np.ndarray,
        b: np.ndarray,
        approximations: List[Tuple[np.ndarray, np.ndarray]],
        *,
        figsize: Tuple[int, int] = (12, 8),
        iteration_axis_label: str = "Iteration",
        y_label: str = "X",
        z_label: str = "Y",
        title: str = "Polygon Evolution: Original (Blue) vs Approximations (Red)",
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the evolution of polygon approximations in 3D.

        Parameters
        ----------
        A, b : define the original polygon as Ax <= b (A: (n,2), b: (n,) or (n,1))
        approximations : list of (C, d) each defining Cx <= d for an iteration
        figsize : figure size
        fig, ax : optionally pass an existing 3D axis to draw on
        show : call plt.show()
        save_path : if provided, saves the figure to this path

        Returns
        -------
        (fig, ax)
        """

        # Convert constraints to vertices
        original_vertices = self.solve_inequalities_to_vertices(A, b)
        if original_vertices is None:
            raise ValueError("Original polygon constraints produce no feasible polygon.")

        # Pre-compute all iteration vertices (some may be None; we skip those)
        approx_vertices_list: List[Optional[np.ndarray]] = []
        for (C, d) in approximations:
            approx_vertices_list.append(self.solve_inequalities_to_vertices(C, d))

        num_iterations = len(approximations)
        x_start = 0
        x_end = num_iterations - 1 if num_iterations > 1 else 1

        # Prepare figure/axes
        if fig is None or ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        # Clean axis look
        if self.axis_off:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.axis("off")

        # ---- Original polygon tube (constant prism from x=0 to x=end) ----
        side_surfaces = []
        m0 = len(original_vertices)
        for j in range(m0):
            nj = (j + 1) % m0
            side_face = [
                (x_start, original_vertices[j, 0], original_vertices[j, 1]),
                (x_start, original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end,   original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end,   original_vertices[j, 0], original_vertices[j, 1]),
            ]
            side_surfaces.append(side_face)

        if side_surfaces:
            side_collection = Poly3DCollection(
                side_surfaces,
                alpha=self.tube_alpha,
                facecolor=self.colors["original_tube_face"],
                edgecolor=self.colors["original_tube_edge"],
            )
            ax.add_collection3d(side_collection)

        # ---- Approximation evolution surfaces + end faces ----
        approx_surfaces = []       # transition tubes
        approx_end_faces = []      # cross-sections at each iteration
        original_end_faces = []    # cross-sections of original at each iteration

        for i, verts in enumerate(approx_vertices_list):
            if verts is None or len(verts) < 3:
                # Skip infeasible or degenerate iterations
                continue

            x_pos = i

            # Connect to previous valid iteration to form transition surfaces
            prev_idx = i - 1
            while prev_idx >= 0 and (approx_vertices_list[prev_idx] is None or len(approx_vertices_list[prev_idx]) < 3):
                prev_idx -= 1

            if prev_idx >= 0:
                prev_verts = approx_vertices_list[prev_idx]
                assert prev_verts is not None
                # pair up by index (fallback: min length)
                min_len = min(len(verts), len(prev_verts))
                for j in range(min_len):
                    nj = (j + 1) % min_len
                    transition_face = [
                        (prev_idx, prev_verts[j, 0], prev_verts[j, 1]),
                        (prev_idx, prev_verts[nj, 0], prev_verts[nj, 1]),
                        (x_pos,    verts[nj, 0],     verts[nj, 1]),
                        (x_pos,    verts[j, 0],      verts[j, 1]),
                    ]
                    approx_surfaces.append(transition_face)

            # Current cross-section faces (approx & original)
            approx_end_faces.append([(x_pos, verts[j, 0], verts[j, 1]) for j in range(len(verts))])
            original_end_faces.append([(x_pos, original_vertices[j, 0], original_vertices[j, 1]) for j in range(m0)])

        # Add collections
        if approx_surfaces:
            approx_tube_collection = Poly3DCollection(
                approx_surfaces,
                alpha=self.tube_alpha,
                facecolor=self.colors["approx_tube_face"],
                edgecolor=self.colors["approx_tube_edge"],
            )
            ax.add_collection3d(approx_tube_collection)

        if original_end_faces:
            original_faces_collection = Poly3DCollection(
                original_end_faces,
                alpha=self.end_face_alpha,
                facecolor=self.colors["original_end_face"],
                edgecolor=self.colors["original_end_edge"],
            )
            ax.add_collection3d(original_faces_collection)
        if approx_end_faces:
            approx_sections_collection = Poly3DCollection(
                approx_end_faces,
                alpha=self.end_face_alpha,
                facecolor=self.colors["approx_end_face"],
                edgecolor=self.colors["approx_end_edge"],
            )
            ax.add_collection3d(approx_sections_collection)

        # ---- Axis limits (equal-ish aspect) ----
        all_vertices = [original_vertices] + [v for v in approx_vertices_list if v is not None]
        all_points = np.vstack(all_vertices)
        y_min, y_max = all_points[:, 0].min(), all_points[:, 0].max()
        z_min, z_max = all_points[:, 1].min(), all_points[:, 1].max()
        span = max(y_max - y_min, z_max - z_min)
        margin = 0.1 * (span if span > 0 else 1.0)

        ax.set_xlim(0, max(1, num_iterations - 1))
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)

        # Labels / title (only if axis isn't fully off)
        if not self.axis_off:
            ax.set_xlabel(iteration_axis_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_zlabel(z_label, fontsize=12)
            ax.set_title(title, fontsize=14)
        else:
            # Even when axis is off, a title can still be useful:
            ax.set_title(title, fontsize=14)

        plt.tight_layout()
        ax.view_init(elev=self.elev, azim=self.azim)
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight", transparent=True)
        if show:
            plt.show()

        return fig, ax
    @staticmethod
    def generate_ellipse_vertices(Sigma: np.ndarray, num_points: int = 100) -> np.ndarray:
        """
        Generate vertices for an ellipse defined by the equation x' Σ x <= 1.
        Sigma is a 2x2 matrix defining the ellipse.

        Parameters
        ----------
        Sigma : 2x2 matrix defining the ellipse shape.
        num_points : Number of points to generate for the ellipse's perimeter.

        Returns
        -------
        vertices : (num_points, 2) array of ellipse vertices.
        """
        # Generate points on the unit circle (x^2 + y^2 = 1)
        theta = np.linspace(0, 2 * np.pi, num_points)
        circle_points = np.column_stack([np.cos(theta), np.sin(theta)])

        # Transform the circle points by Sigma to get the ellipse
        eigvals, eigvecs = np.linalg.eig(Sigma)
        ellipse_points = eigvecs @ np.diag(np.sqrt(1 / eigvals)) @ circle_points.T
        return ellipse_points.T

    def plot_ellipse_evolution(
            self,
            Sigma: np.ndarray,
            approximations: List[Tuple[np.ndarray, np.ndarray]],
            *,
            figsize: Tuple[int, int] = (12, 8),
            iteration_axis_label: str = "Iteration",
            y_label: str = "X",
            z_label: str = "Y",
            title: str = "Ellipse Evolution: Original (Blue) vs Approximations (Red)",
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            show: bool = True,
            save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the evolution of ellipse approximations in 3D.

        Parameters
        ----------
        Sigma : 2x2 matrix defining the original ellipse (x' Σ x <= 1).
        approximations : list of (C, d) tuples, each defining an approximation for an iteration.
        figsize : figure size
        fig, ax : optionally pass an existing 3D axis to draw on
        show : call plt.show()
        save_path : if provided, saves the figure to this path

        Returns
        -------
        (fig, ax)
        """
        # Generate the original ellipse vertices
        original_vertices = self.generate_ellipse_vertices(Sigma)

        # Pre-compute all iteration vertices (some may be None; we skip those)
        approx_vertices_list: List[Optional[np.ndarray]] = []
        for (C, d) in approximations:
            approx_vertices_list.append(self.solve_inequalities_to_vertices(C, d))

        num_iterations = len(approximations)
        x_start = 0
        x_end = num_iterations - 1 if num_iterations > 1 else 1

        # Prepare figure/axes
        if fig is None or ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        # Clean axis look
        if self.axis_off:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.axis("off")

        # ---- Original ellipse tube (constant prism from x=0 to x=end) ----
        ellipse_surfaces = []
        m0 = len(original_vertices)
        for j in range(m0):
            nj = (j + 1) % m0
            ellipse_surfaces.append([
                (x_start, original_vertices[j, 0], original_vertices[j, 1]),
                (x_start, original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end, original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end, original_vertices[j, 0], original_vertices[j, 1]),
            ])
        ax.add_collection3d(Poly3DCollection(
            ellipse_surfaces,
            alpha=self.tube_alpha,
            facecolor=self.colors["original_tube_face"],
            edgecolor=None,
        ))

        # ---- Approximation evolution surfaces ----
        approx_surfaces = []
        approx_end_faces = []
        original_end_faces = []
        for i, verts in enumerate(approx_vertices_list):
            if verts is None or len(verts) < 3:
                # Skip infeasible or degenerate iterations
                continue

            x_pos = i

            # Connect to previous valid iteration to form transition surfaces
            prev_idx = i - 1
            while prev_idx >= 0 and (approx_vertices_list[prev_idx] is None or len(approx_vertices_list[prev_idx]) < 3):
                prev_idx -= 1

            if prev_idx >= 0:
                prev_verts = approx_vertices_list[prev_idx]
                assert prev_verts is not None
                # pair up by index (fallback: min length)
                min_len = min(len(verts), len(prev_verts))
                for j in range(min_len):
                    nj = (j + 1) % min_len
                    transition_face = [
                        (prev_idx, prev_verts[j, 0], prev_verts[j, 1]),
                        (prev_idx, prev_verts[nj, 0], prev_verts[nj, 1]),
                        (x_pos,    verts[nj, 0],     verts[nj, 1]),
                        (x_pos,    verts[j, 0],      verts[j, 1]),
                    ]
                    approx_surfaces.append(transition_face)

            # Current cross-section faces (approx & original)
            approx_end_faces.append([(x_pos, verts[j, 0], verts[j, 1]) for j in range(len(verts))])
            original_end_faces.append([(x_pos, original_vertices[j, 0], original_vertices[j, 1]) for j in range(m0)])


        if original_end_faces:
            original_faces_collection = Poly3DCollection(
                original_end_faces,
                alpha=self.end_face_alpha,
                facecolor=self.colors["original_end_face"],
                edgecolor=self.colors["original_end_edge"],
            )
            ax.add_collection3d(original_faces_collection)
        # Add collections for approximations
        if approx_surfaces:
            ax.add_collection3d(Poly3DCollection(
                approx_surfaces,
                alpha=self.tube_alpha,
                facecolor=self.colors["approx_tube_face"],
                edgecolor=self.colors["approx_tube_edge"],
            ))

        if approx_end_faces:
            ax.add_collection3d(Poly3DCollection(
                approx_end_faces,
                alpha=self.end_face_alpha,
                facecolor=self.colors["approx_end_face"],
                edgecolor=self.colors["approx_end_edge"],
            ))

        # ---- Axis limits ----
        all_vertices = [original_vertices] + [v for v in approx_vertices_list if v is not None]
        all_points = np.vstack(all_vertices)
        y_min, y_max = all_points[:, 0].min(), all_points[:, 0].max()
        z_min, z_max = all_points[:, 1].min(), all_points[:, 1].max()
        span = max(y_max - y_min, z_max - z_min)
        margin = 0.1 * (span if span > 0 else 1.0)

        ax.set_xlim(0, max(1, num_iterations - 1))
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)

        # Labels / title
        if not self.axis_off:
            ax.set_xlabel(iteration_axis_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_zlabel(z_label, fontsize=12)
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(title, fontsize=14)

        plt.tight_layout()
        ax.view_init(elev=self.elev, azim=self.azim)
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight", transparent=True)

        if show:
            plt.show()

        return fig, ax

    @staticmethod
    def generate_circle_region_vertices(center1: Tuple[float, float], radius1: float,
                                        center2: Tuple[float, float], radius2: float,
                                        num_points: int = 200) -> np.ndarray:
        """
        Generate vertices for a non-convex region: circle1 minus the intersection with circle2.
        This creates the region inside circle1 but outside circle2.

        Parameters
        ----------
        center1 : (x, y) center of the first circle (the main region)
        radius1 : radius of the first circle
        center2 : (x, y) center of the second circle (the subtracted region)
        radius2 : radius of the second circle
        num_points : Number of points to generate for the boundary

        Returns
        -------
        vertices : (n, 2) array of boundary vertices, or None if no valid region
        """
        x1, y1 = center1
        x2, y2 = center2

        # Distance between centers
        d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Check if circles intersect
        if d >= radius1 + radius2:
            # No intersection, return full circle1
            theta = np.linspace(0, 2 * np.pi, num_points)
            vertices = np.column_stack([
                x1 + radius1 * np.cos(theta),
                y1 + radius1 * np.sin(theta)
            ])
            return vertices

        if d + radius1 <= radius2:
            # Circle1 completely inside circle2, no region left
            return None

        if d + radius2 <= radius1:
            # Circle2 completely inside circle1, create annulus-like region
            # Outer boundary: circle1, Inner boundary: circle2 (reversed)
            theta1 = np.linspace(0, 2 * np.pi, num_points // 2)
            outer_boundary = np.column_stack([
                x1 + radius1 * np.cos(theta1),
                y1 + radius1 * np.sin(theta1)
            ])

            theta2 = np.linspace(2 * np.pi, 0, num_points // 2)  # Reversed for inner hole
            inner_boundary = np.column_stack([
                x2 + radius2 * np.cos(theta2),
                y2 + radius2 * np.sin(theta2)
            ])

            # Connect outer to inner with a thin bridge to make it simply connected
            bridge_point1 = outer_boundary[0]
            bridge_point2 = inner_boundary[0]

            vertices = np.vstack([outer_boundary, [bridge_point1], inner_boundary, [bridge_point2]])
            return vertices

        # Circles partially intersect - find intersection points
        a = (radius1 ** 2 - radius2 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(radius1 ** 2 - a ** 2)

        # Intersection point on line between centers
        px = x1 + a * (x2 - x1) / d
        py = y1 + a * (y2 - y1) / d

        # Two intersection points
        intersect1 = np.array([px + h * (y2 - y1) / d, py - h * (x2 - x1) / d])
        intersect2 = np.array([px - h * (y2 - y1) / d, py + h * (x2 - x1) / d])

        # Find angles for intersection points relative to each circle center
        angle1_c1 = np.arctan2(intersect1[1] - y1, intersect1[0] - x1)
        angle2_c1 = np.arctan2(intersect2[1] - y1, intersect2[0] - x1)
        angle1_c2 = np.arctan2(intersect1[1] - y2, intersect1[0] - x2)
        angle2_c2 = np.arctan2(intersect2[1] - y2, intersect2[0] - x2)

        # Normalize angles to [0, 2π]
        angle1_c1 = angle1_c1 % (2 * np.pi)
        angle2_c1 = angle2_c1 % (2 * np.pi)
        angle1_c2 = angle1_c2 % (2 * np.pi)
        angle2_c2 = angle2_c2 % (2 * np.pi)

        # Determine which arc of circle1 to keep (the one outside circle2)
        # Test the midpoint of each possible arc
        if angle2_c1 < angle1_c1:
            angle2_c1 += 2 * np.pi

        # Test arc from angle1 to angle2
        test_angle1 = (angle1_c1 + angle2_c1) / 2
        test_point1 = np.array([x1 + radius1 * np.cos(test_angle1),
                                y1 + radius1 * np.sin(test_angle1)])
        test_dist1 = np.sqrt((test_point1[0] - x2) ** 2 + (test_point1[1] - y2) ** 2)

        # Test complementary arc
        test_angle2 = test_angle1 + np.pi
        test_point2 = np.array([x1 + radius1 * np.cos(test_angle2),
                                y1 + radius1 * np.sin(test_angle2)])
        test_dist2 = np.sqrt((test_point2[0] - x2) ** 2 + (test_point2[1] - y2) ** 2)

        if test_dist1 > radius2:
            # Keep arc from angle1 to angle2
            theta_c1 = np.linspace(angle1_c1, angle2_c1, num_points // 3)
            arc1_points = np.column_stack([
                x1 + radius1 * np.cos(theta_c1),
                y1 + radius1 * np.sin(theta_c1)
            ])
        else:
            # Keep complementary arc
            if angle1_c1 > angle2_c1:
                theta_c1 = np.linspace(angle2_c1, angle1_c1, num_points // 3)
            else:
                theta_c1 = np.concatenate([
                    np.linspace(angle2_c1, 2 * np.pi, num_points // 6),
                    np.linspace(0, angle1_c1, num_points // 6)
                ])
            arc1_points = np.column_stack([
                x1 + radius1 * np.cos(theta_c1),
                y1 + radius1 * np.sin(theta_c1)
            ])

        # Now add the arc of circle2 that forms the boundary (from intersect2 to intersect1)
        # This is the key fix - we need the arc of circle2 that's inside circle1
        if angle1_c2 < angle2_c2:
            # Check which direction gives us the arc inside circle1
            mid_angle = (angle1_c2 + angle2_c2) / 2
            mid_point = np.array([x2 + radius2 * np.cos(mid_angle),
                                  y2 + radius2 * np.sin(mid_angle)])
            mid_dist = np.sqrt((mid_point[0] - x1) ** 2 + (mid_point[1] - y1) ** 2)

            if mid_dist < radius1:
                # Arc from angle2 to angle1 (clockwise)
                theta_c2 = np.linspace(angle2_c2, angle1_c2, num_points // 3)
            else:
                # Arc from angle1 to angle2 (counter-clockwise)
                theta_c2 = np.linspace(angle1_c2, angle2_c2, num_points // 3)
        else:
            # angle2_c2 < angle1_c2, need to check which way around
            mid_angle1 = (angle2_c2 + angle1_c2) / 2
            mid_angle2 = mid_angle1 + np.pi

            mid_point1 = np.array([x2 + radius2 * np.cos(mid_angle1),
                                   y2 + radius2 * np.sin(mid_angle1)])
            mid_dist1 = np.sqrt((mid_point1[0] - x1) ** 2 + (mid_point1[1] - y1) ** 2)

            if mid_dist1 < radius1:
                theta_c2 = np.linspace(angle2_c2, angle1_c2, num_points // 3)
            else:
                # Go the long way around
                theta_c2 = np.concatenate([
                    np.linspace(angle1_c2, 2 * np.pi, num_points // 6),
                    np.linspace(0, angle2_c2, num_points // 6)
                ])

        arc2_points = np.column_stack([
            x2 + radius2 * np.cos(theta_c2),
            y2 + radius2 * np.sin(theta_c2)
        ])

        # Combine the arcs: arc1 + arc2 to form the complete boundary
        vertices = np.vstack([arc1_points, arc2_points])

        return vertices

    def plot_circle_region_evolution(
            self,
            center1: Tuple[float, float],
            radius1: float,
            center2: Tuple[float, float],
            radius2: float,
            approximations: List[Tuple[np.ndarray, np.ndarray]],
            *,
            figsize: Tuple[int, int] = (12, 8),
            iteration_axis_label: str = "Iteration",
            y_label: str = "X",
            z_label: str = "Y",
            title: str = "Circle Region Evolution: Original (Blue) vs Approximations (Red)",
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            show: bool = True,
            save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the evolution of circle region approximations in 3D.
        The original region is defined as circle1 minus the intersection with circle2.

        Parameters
        ----------
        center1 : (x, y) center of the main circle
        radius1 : radius of the main circle
        center2 : (x, y) center of the subtracted circle
        radius2 : radius of the subtracted circle
        approximations : list of (C, d) tuples, each defining a polyhedral approximation
        figsize : figure size
        fig, ax : optionally pass an existing 3D axis to draw on
        show : call plt.show()
        save_path : if provided, saves the figure to this path

        Returns
        -------
        (fig, ax)
        """
        # Generate the original circle region vertices
        original_vertices = self.generate_circle_region_vertices(center1, radius1, center2, radius2)
        if original_vertices is None:
            raise ValueError("Circle region constraints produce no feasible region.")

        # Pre-compute all iteration vertices (some may be None; we skip those)
        approx_vertices_list: List[Optional[np.ndarray]] = []
        for (C, d) in approximations:
            approx_vertices_list.append(self.solve_inequalities_to_vertices(C, d))

        num_iterations = len(approximations)
        x_start = 0
        x_end = num_iterations - 1 if num_iterations > 1 else 1

        # Prepare figure/axes
        if fig is None or ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        # Clean axis look
        if self.axis_off:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.axis("off")

        # ---- Original circle region tube (constant prism from x=0 to x=end) ----
        region_surfaces = []
        m0 = len(original_vertices)
        for j in range(m0):
            nj = (j + 1) % m0
            region_surfaces.append([
                (x_start, original_vertices[j, 0], original_vertices[j, 1]),
                (x_start, original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end, original_vertices[nj, 0], original_vertices[nj, 1]),
                (x_end, original_vertices[j, 0], original_vertices[j, 1]),
            ])

        if region_surfaces:
            ax.add_collection3d(Poly3DCollection(
                region_surfaces,
                alpha=self.tube_alpha,
                facecolor=self.colors["original_tube_face"],
                edgecolor=None,
            ))

        # ---- Approximation evolution surfaces ----
        approx_surfaces = []
        approx_end_faces = []
        original_end_faces = []

        for i, verts in enumerate(approx_vertices_list):
            if verts is None or len(verts) < 3:
                # Skip infeasible or degenerate iterations
                continue

            x_pos = i

            # Connect to previous valid iteration to form transition surfaces
            prev_idx = i - 1
            while prev_idx >= 0 and (approx_vertices_list[prev_idx] is None or len(approx_vertices_list[prev_idx]) < 3):
                prev_idx -= 1

            if prev_idx >= 0:
                prev_verts = approx_vertices_list[prev_idx]
                assert prev_verts is not None
                # pair up by index (fallback: min length)
                min_len = min(len(verts), len(prev_verts))
                for j in range(min_len):
                    nj = (j + 1) % min_len
                    transition_face = [
                        (prev_idx, prev_verts[j, 0], prev_verts[j, 1]),
                        (prev_idx, prev_verts[nj, 0], prev_verts[nj, 1]),
                        (x_pos, verts[nj, 0], verts[nj, 1]),
                        (x_pos, verts[j, 0], verts[j, 1]),
                    ]
                    approx_surfaces.append(transition_face)

            # Current cross-section faces (approx & original)
            approx_end_faces.append([(x_pos, verts[j, 0], verts[j, 1]) for j in range(len(verts))])
            original_end_faces.append([(x_pos, original_vertices[j, 0], original_vertices[j, 1]) for j in range(m0)])

        # Add collections
        if original_end_faces:
            original_faces_collection = Poly3DCollection(
                original_end_faces,
                alpha=self.end_face_alpha,
                facecolor=self.colors["original_end_face"],
                edgecolor=self.colors["original_end_edge"],
            )
            ax.add_collection3d(original_faces_collection)

        if approx_surfaces:
            ax.add_collection3d(Poly3DCollection(
                approx_surfaces,
                alpha=self.tube_alpha,
                facecolor=self.colors["approx_tube_face"],
                edgecolor=self.colors["approx_tube_edge"],
            ))

        if approx_end_faces:
            ax.add_collection3d(Poly3DCollection(
                approx_end_faces,
                alpha=self.end_face_alpha,
                facecolor=self.colors["approx_end_face"],
                edgecolor=self.colors["approx_end_edge"],
            ))

        # ---- Axis limits ----
        all_vertices = [original_vertices] + [v for v in approx_vertices_list if v is not None]
        all_points = np.vstack(all_vertices)
        y_min, y_max = all_points[:, 0].min(), all_points[:, 0].max()
        z_min, z_max = all_points[:, 1].min(), all_points[:, 1].max()
        span = max(y_max - y_min, z_max - z_min)
        margin = 0.1 * (span if span > 0 else 1.0)

        ax.set_xlim(0, max(1, num_iterations - 1))
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)

        # Labels / title
        if not self.axis_off:
            ax.set_xlabel(iteration_axis_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_zlabel(z_label, fontsize=12)
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(title, fontsize=14)

        plt.tight_layout()
        ax.view_init(elev=self.elev, azim=self.azim)
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight", transparent=True)

        if show:
            plt.show()

        return fig, ax
def plot_errors(error_feas, error_opt, n_train,interporation = True):
    from scipy.interpolate import make_interp_spline

    # Original x values (assuming iterations are evenly spaced up to 1000)
    iterations = np.linspace(0, n_train, len(error_feas))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 3))

    if interporation:
        # Create new x values for smooth curve
        iterations_smooth = np.linspace(0, n_train, 100)  # 300 points for smoothness

        # Create spline interpolations
        feas_spline = make_interp_spline(iterations, np.log10(np.array(error_feas)+1e-10), k=3)
        opt_spline = make_interp_spline(iterations, np.log10(np.array(error_opt)+1e-10), k=3)

        # Evaluate splines at new points
        feas_smooth = feas_spline(iterations_smooth)
        opt_smooth = opt_spline(iterations_smooth)

        # Plot smooth curves without markers
        ax.semilogy(iterations_smooth, np.power(10,feas_smooth) , '-', color='#D95F41', linewidth=2)
        ax.semilogy(iterations_smooth, np.power(10,opt_smooth), '-', color='#3B75AF', linewidth=2)
    else:
        # Plot smooth curves without markers
        ax.semilogy(iterations, error_feas, '-', color='#D95F41', linewidth=2)
        ax.semilogy(iterations, error_opt, '-', color='#3B75AF', linewidth=2)
    # Set axis limits and ticks
    ax.set_xlim(0, n_train)
    ax.set_xticks(np.arange(0, n_train+1, int(n_train/2)))

    fontsize = 30
    ax.set_xticklabels(np.arange(0, n_train+1, int(n_train/2)), fontsize=fontsize )
    ax.set_ylim(1e-4, 1e0)

    # Custom y-axis ticks and labels
    yticks = [1e0, 1e-2, 1e-4]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['1e0', '1e-2', '1e-4'], fontsize=fontsize )
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    # Remove borders and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.grid(False)

    # Labels and legend
    ax.set_xlabel('Iteration', fontsize=fontsize )
    ax.set_ylabel('Error', fontsize=fontsize )

    plt.tight_layout()
    plt.show()

def create_gif(image_folder, output_gif, duration=300, loop=0):
    """
    使用Pillow生成GIF动画

    参数:
        image_folder: 图片目录路径
        output_gif: 输出GIF文件名
        duration: 每帧持续时间(毫秒)
        loop: 循环次数(0表示无限循环)
    """
    # 获取按数字排序的图片文件
    images = []
    files = sorted([f for f in os.listdir(image_folder)
                    if f.startswith('step') and f.endswith('.png')],
                   key=lambda x: int(x[4:-4]))

    # 加载所有图片帧
    frames = []
    for file in files:
        frame = Image.open(os.path.join(image_folder, file))
        frames.append(frame.copy())
        frame.close()  # 显式关闭文件

    # 保存为GIF（优化处理）
    if len(frames) > 0:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
            optimize=True,  # 启用优化
            # disposal=2  # 每帧恢复背景
        )
        print(f"成功生成GIF: {output_gif} (共{len(frames)}帧)")
    else:
        print("未找到有效图片帧！")

# import seaborn as sns
class ErrorVisualizer:
    def __init__(self):
        """
        独立的误差可视化工具
        跟踪记录两种误差：
        - error_feas: 可行性误差
        - error_opt: 最优性误差
        """
        self.error_history = {
            'iterations': [],
            'error_feas': [],  # 存储每次迭代的可行性误差列表
            'error_opt': []  # 存储每次迭代的最优性误差列表
        }

    def compute_errors(self, model, num_sample=50):
        """
        计算并记录误差（原count_error函数的功能）

        参数:
            model: 需要包含以下方法：
                - optimize_direction()
                - project()
            num_sample: 采样次数
        """
        error_feas = []
        error_opt = []

        for _ in range(num_sample):
            c = np.random.randn(model.dim)
            # 计算可行性误差
            x_apx = model.optimize_direction(c, in_approx=True)
            x_org = model.project(x_apx) if x_apx is not None else None
            if x_org is not None:
                error_feas.append(np.sum((x_apx - x_org)  ** 2))
            # 计算最优性误差
            x_org = model.optimize_direction(c)
            x_apx = model.project(x_org, to_approx=True) if x_org is not None else None
            if x_apx is not None:
                error_opt.append(np.sum((x_apx - x_org)  ** 2))

        self.error_history['iterations'].append(model._iter)
        self.error_history['error_feas'].append(np.array(error_feas))
        self.error_history['error_opt'].append(np.array(error_opt))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def plot_dual_violin(self, save_path=None, figsize=(12, 8)):
        """
        精确连接均值点的小提琴图版本
        """
        plt.figure(figsize=figsize)

        iterations = self.error_history['iterations']
        # iterations[0] = 0
        feas_data = self.error_history['error_feas']
        opt_data = self.error_history['error_opt']

        display_step = max(1, len(iterations) // 10)
        display_indices = range(0, len(iterations), display_step)
        display_iters = [iterations[i]-1 for i in display_indices]

        positions = []
        all_data = []
        for idx in display_indices:
            positions.extend([idx * 2, idx * 2 + 0.8])
            all_data.append(feas_data[idx])
            all_data.append(opt_data[idx])

        data_for_plot = []
        for i in range(len(positions)):
            data_for_plot.append(all_data[i])
        plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
        parts = plt.violinplot(
            dataset=data_for_plot,
            positions=positions,
            widths=0.6,
            showmeans=True,
            showextrema=True,
            # showmedians=True
        )

        colors = ['#3498db', '#e74c3c']
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(colors[i % 2])
            body.set_alpha(0.6)
            body.set_edgecolor('black')
            body.set_linewidth(1)

        feas_means_x, feas_means_y = [], []
        opt_means_x, opt_means_y = [], []

        for i, (x, y_line) in enumerate(zip(positions, parts['cmeans'].get_segments())):
            x_coord = (y_line[0][0] + y_line[1][0]) / 2
            y_coord = (y_line[0][1] + y_line[1][1]) / 2
            plt.plot(x_coord, y_coord, marker='D', color='black', markersize=8, markerfacecolor='white',
                     linestyle='None')
            if i % 2 == 0:
                feas_means_x.append(x)
                feas_means_y.append(y_coord)
            else:
                opt_means_x.append(x)
                opt_means_y.append(y_coord)

        plt.plot(feas_means_x, feas_means_y, color=colors[0], linestyle='-', linewidth=2, alpha=0.7,
                 label='Feas Mean Curve')
        plt.plot(opt_means_x, opt_means_y, color=colors[1], linestyle='--', linewidth=2, alpha=0.7,
                 label='Opt Mean Curve')

        x_ticks = [(feas_means_x[i] + opt_means_x[i]) / 2 for i in range(len(feas_means_x))]
        plt.xticks(x_ticks, [f'{iter}' for iter in display_iters])

        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Error Value', fontsize=12)
        plt.title('Error Distribution Evolution (Violin)', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')

        legend_elements = [
            Line2D([0], [0], color=colors[0], lw=4, label='Feasibility error'),
            Line2D([0], [0], color=colors[1], lw=4, label='Optimality error'),
            Line2D([0], [0], color=colors[0], lw=2, linestyle='-', label='Feas mean curve'),
            Line2D([0], [0], color=colors[1], lw=2, linestyle='--', label='Opt mean curve'),
            # Line2D([0], [0], color='black', lw=2, label='Median'),
            Line2D([0], [0], marker='D', color='black', label='Mean',
                   markersize=8, linestyle='None', markerfacecolor='white')
        ]

        plt.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
        plt.subplots_adjust(right=0.75)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()

    def plot_dual_boxplot(self, save_path=None, figsize=(12, 8)):
        """
        精确连接均值点的箱线图版本
        修复内容：
        1. 趋势线准确连接各自箱线图的均值标记
        2. 优化图例和标签显示
        """
        plt.figure(figsize=figsize)

        # 数据准备
        iterations = self.error_history['iterations']
        feas_data = self.error_history['error_feas']
        opt_data = self.error_history['error_opt']

        # 自动调整显示密度
        display_step = max(1, len(iterations) // 10)
        display_indices = range(0, len(iterations), display_step)
        display_iters = [iterations[i] for i in display_indices]

        # 创建绘图位置（每组两个箱体：feas和opt）
        positions = []
        for idx in display_indices:
            positions.extend([idx * 2, idx * 2 + 0.8])  # feas在左，opt在右

        # 合并数据（交替存储feas和opt）
        all_data = []
        for idx in display_indices:
            all_data.append(feas_data[idx])
            all_data.append(opt_data[idx])

        # 配置箱线图属性
        boxprops = dict(linewidth=1.5)
        medianprops = dict(linewidth=2, color='black')
        meanprops = dict(marker='D', markersize=8,
                         markerfacecolor='white',
                         markeredgecolor='black')
        flierprops = dict(marker='o', markersize=5,
                          markerfacecolor='none',
                          markeredgecolor='gray', alpha=0.6)

        # 绘制箱线图
        box = plt.boxplot(
            all_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            boxprops=boxprops,
            medianprops=medianprops,
            meanprops=meanprops,
            flierprops=flierprops,
            whiskerprops=boxprops,
            capprops=boxprops
        )

        # 设置箱体颜色
        colors = ['#3498db', '#e74c3c']
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor(colors[i % 2])
            patch.set_alpha(0.8)

        # =====================================
        # 关键修复：精确连接各自的均值点
        # =====================================

        # 获取所有均值点的坐标（box['means']返回的Artist对象列表）
        means = box['means']

        # 分离feas和opt的均值点坐标
        feas_means_x = []
        feas_means_y = []
        opt_means_x = []
        opt_means_y = []

        for i, mean in enumerate(means):
            x = mean.get_xdata()[0]  # 均值标记的x坐标
            y = mean.get_ydata()[0]  # 均值标记的y坐标
            if i % 2 == 0:  # 偶数索引是feas
                feas_means_x.append(x)
                feas_means_y.append(y)
            else:  # 奇数索引是opt
                opt_means_x.append(x)
                opt_means_y.append(y)

        # 绘制趋势线（连接各自的实际均值点）
        plt.plot(feas_means_x, feas_means_y,
                 color=colors[0], linestyle='-',
                 linewidth=2, alpha=0.7, label='Feas Mean Curve')

        plt.plot(opt_means_x, opt_means_y,
                 color=colors[1], linestyle='--',
                 linewidth=2, alpha=0.7, label='Opt Mean Curve')

        # =====================================
        # 坐标轴和图例设置
        # =====================================

        # 设置x轴刻度（每组箱线图的中间位置）
        x_ticks = [(feas_means_x[i] + opt_means_x[i]) / 2 for i in range(len(feas_means_x))]
        plt.xticks(
            x_ticks,
            [f' {iter}' for iter in display_iters],
            # rotation=45
        )

        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Error Value', fontsize=12)
        plt.title('Error Distribution Evolution', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')

        # 构建完整图例系统
        from matplotlib.lines import Line2D
        legend_elements = [
            # 误差类型图例
            Line2D([0], [0], color=colors[0], lw=4, label='Feasibility error'),
            Line2D([0], [0], color=colors[1], lw=4, label='Optimality error'),

            # 趋势线图例
            Line2D([0], [0], color=colors[0], lw=2, linestyle='-', label='Feas mean curve'),
            Line2D([0], [0], color=colors[1], lw=2, linestyle='--', label='Opt mean curve'),

            # 统计量图例
            Line2D([0], [0], color='black', lw=2, label='Median'),
            Line2D([0], [0], marker='D', color='black', label='Mean',
                   markersize=8, linestyle='None', markerfacecolor='white'),
            Line2D([0], [0], marker='o', color='gray', label='Outliers',
                   markersize=8, linestyle='None', markerfacecolor='none')
        ]

        plt.legend(handles=legend_elements,
                   loc='upper right',
                   fontsize=12,
                   framealpha=0.9)

        plt.subplots_adjust(right=0.75)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_2d_projection(self, true_model, A_pred, b_pred, t1=12, t2=13,
                           num_points=36, var_name='P_AGG', save_path=None, title=None,
                           solver_name='gurobi'):
        """
        绘制 24 维可行域在 t1 和 t2 两个时间维度的 2D 投影对比图。

        参数:
            true_model: Pyomo 真实模型对象
            A_pred: 预测出的 A 矩阵 (numpy array, 形状 [m, n])
            b_pred: 预测出的 b 向量 (numpy array, 形状 [m])
            t1, t2: 要展示的两个时间索引 (默认中午12点和下午13点)
            num_points: 采样的方向数量 (默认 36 个方向，即每 10 度探测一次)
            var_name: Pyomo 模型中聚合功率的变量名称
            save_path: 图片保存路径
            solver_name: Pyomo求解器名称，默认 gurobi
        """
        import pyomo.environ as pyo
        import math
        from scipy.optimize import linprog

        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        true_points = []
        pred_points = []

        solver = pyo.SolverFactory(solver_name)

        # 获取 Pyomo 中的决策变量
        agg_var = getattr(true_model, var_name)
        n_dim = A_pred.shape[1]

        print(f"开始计算可行域 2D 切面 (维度 {t1} 与 {t2})，共需求解 {num_points * 2} 个 LP 问题...")

        for theta in angles:
            c1, c2 = math.cos(theta), math.sin(theta)

            # ==========================================
            # 1. 计算真实边界 (求解 Pyomo 模型)
            # ==========================================
            # 如果已经存在用于射线探测的目标函数，先删除它
            if hasattr(true_model, 'ray_obj'):
                true_model.del_component(true_model.ray_obj)

            # 最大化当前射线方向
            true_model.ray_obj = pyo.Objective(
                expr=c1 * agg_var[t1] + c2 * agg_var[t2],
                sense=pyo.maximize
            )

            res = solver.solve(true_model, tee=False)
            if res.solver.termination_condition in [pyo.TerminationCondition.optimal,
                                                    pyo.TerminationCondition.feasible]:
                true_points.append([pyo.value(agg_var[t1]), pyo.value(agg_var[t2])])
            else:
                print(f"  [警告] 真实模型在角度 {math.degrees(theta):.1f}° 求解失败")

            # ==========================================
            # 2. 计算预测边界 (求解 A_pred * x <= b_pred)
            # ==========================================
            # scipy.linprog 求的是 min c^T x，所以要求 max 需要将 c 取负号
            c_vec = np.zeros(n_dim)
            c_vec[t1] = -c1
            c_vec[t2] = -c2

            # 求解线性规划
            res_pred = linprog(c_vec, A_ub=A_pred, b_ub=b_pred, bounds=(None, None), method='highs')
            if res_pred.success:
                pred_points.append([res_pred.x[t1], res_pred.x[t2]])
            else:
                print(f"  [警告] 预测模型在角度 {math.degrees(theta):.1f}° 求解失败")

        # ==========================================
        # 3. 闭合并绘制多边形
        # ==========================================
        if len(true_points) > 2 and len(pred_points) > 2:
            true_points = np.array(true_points)
            pred_points = np.array(pred_points)

            # 首尾相连形成闭合圈
            true_points = np.vstack([true_points, true_points[0]])
            pred_points = np.vstack([pred_points, pred_points[0]])

            # 绘制真实可行域（灰色填充 + 黑色边框）
            ax.plot(true_points[:, 0], true_points[:, 1], 'k-', linewidth=2.5, label='True Region (Pyomo)')
            ax.fill(true_points[:, 0], true_points[:, 1], color='gray', alpha=0.3)

            # 绘制预测可行域（红色虚线 + 红色散点）
            ax.plot(pred_points[:, 0], pred_points[:, 1], 'r--', linewidth=2.5, label='Predicted Region (FullNet)')
            ax.scatter(pred_points[:, 0], pred_points[:, 1], color='red', s=30, zorder=5)

            # 美化图表
            ax.set_xlabel(f'Aggregated Power at t={t1} (MW)', fontsize=14)
            ax.set_ylabel(f'Aggregated Power at t={t2} (MW)', fontsize=14)
            if title:
                ax.set_title(title, fontsize=16)
            else:
                ax.set_title(f'Feasible Region 2D Projection (t={t1} vs t={t2})', fontsize=16)

            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(fontsize=12, loc='best')
            ax.tick_params(axis='both', which='major', labelsize=12)

            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, bbox_inches='tight', transparent=False)
                print(f"图片已保存至: {save_path}")

            plt.show()
        else:
            print("计算成功的顶点太少，无法绘图！")

        return fig, ax



# 使用示例
if __name__ == "__main__":
    # type = 'epigraph'
    # create_gif('figures/'+type, type+'.gif', duration=200, loop=1)
    drawer = ShapeDrawer_2D()
    theta = [1,1,1]  # 圆心 (1,1)，半径1
    drawer.plot_circle_regions(
        theta=theta,
        xlim=(-1.5, 1.5),  # 包含两个圆的可视范围
        ylim=(-1.5, 1.5),
        edgecolor='skyblue',
        facecolor='skyblue',  # 区域填充色
        alpha=0.3,  # 透明度
        title='Circle Regions Demo'
    )

    # 添加辅助元素
    drawer.ax.grid(True)
    drawer.ax.axhline(0, color='black', lw=0.5)
    drawer.ax.axvline(0, color='black', lw=0.5)
    drawer.ax.set_aspect('equal')

    # 显示图形
    plt.show()