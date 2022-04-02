from pathlib import Path
import argparse

# https://matplotlib.org/stable/api/tri_api.html
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns
from matplotlib.tri import Triangulation, UniformTriRefiner, LinearTriInterpolator, \
    CubicTriInterpolator
from plotly.figure_factory import create_trisurf
import plotly.graph_objects as go
import plotly.express as px


def plot_time(input_path):
    p = Path(input_path)
    df = get_data(input_path)
    df['total'] = df['datetime_complete'] - df['datetime_start']
    df = df.astype({'total': 'timedelta64[s]'})
    df['fenia.output.time.total'] = df['total'] - df['gmsh.output.time.total']
    ts = ['total',
          'fenia.output.time.total',
          'gmsh.output.time.total',
          'gmsh.output.time.generate',
          'gmsh.output.time.register',
          'gmsh.output.time.boolean',
          'gmsh.output.time.optimize',
          'gmsh.output.time.write',
          'gmsh.output.time.transform',
          'gmsh.output.time.zone',
          'gmsh.output.time.synchronize',
          'gmsh.output.time.structure',
          'gmsh.output.time.pre_unregister',
          'gmsh.output.time.quadrate',
          'gmsh.output.time.refine',
          'gmsh.output.time.size',
          'gmsh.output.time.smooth']
    df = df.melt(var_name='time',
                 id_vars=['number',
                          'gmsh.output.mesh.elements',
                          'gmsh.output.mesh.nodes',
                          'gmsh.input.algorithm.2d',
                          'gmsh.input.algorithm.3d',
                          'gmsh.output.mesh.metric.icn.min',
                          'gmsh.output.mesh.metric.ige.min'
                          ],
                 value_vars=ts,
                 value_name='value')
    df['value'] /= 3600
    sns.set(style="ticks")
    g = sns.catplot(x="value", y="time",
                    kind="boxen", data=df,
                    # scale='width',
                    height=8, aspect=1.777)
    g.set(xscale="log")
    g.set_xlabels(label='hours')
    g.set_yticklabels(labels=[x.replace('.output.time', '') for x in ts])
    plt.grid()  # just add this
    g.savefig(p.with_name('time').with_suffix('.png'))

    fig = px.box(df, x="value", y="time", log_x=True,
                 hover_data=['number',
                             'time',
                             'value',
                             'gmsh.output.mesh.elements',
                             'gmsh.output.mesh.nodes',
                             'gmsh.input.algorithm.2d',
                             'gmsh.input.algorithm.3d',
                             'gmsh.output.mesh.metric.icn.min',
                             'gmsh.output.mesh.metric.ige.min'
                             ], color="time",
                 labels={
                     "value": "hours",
                 }
                 # , box=True

                 )
    fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False)

    fig.write_html(p.with_name(f'time').with_suffix('.html'))


def plot_tri(input_path):
    def tri_by_xyz(p, df, x, y, z, x_label, y_label, cb_label, title):
        xs = df[x].to_numpy()
        ys = df[y].to_numpy()
        zs = df[z].to_numpy()
        plt.figure(figsize=(8, 6))
        # Triangulate
        tri = Triangulation(xs, ys)
        # Write
        fig = create_trisurf(x=tri.x, y=tri.y, z=zs,
                             colormap='Jet',
                             show_colorbar=True,
                             plot_edges=False,
                             # title=title,
                             simplices=tri.triangles)
        fig.update_layout(scene={
            "xaxis_title": x_label,
            "yaxis_title": y_label,
            "zaxis_title": cb_label})
        fig.write_html(p.with_name(f'tri_3d_{z}').with_suffix('.html'))
        plt.tricontourf(tri, zs, levels=10,
                        cmap=cm.jet, alpha=1,
                        norm=plt.Normalize(vmax=zs.max(), vmin=zs.min()))
        plt.plot(xs, ys, 'ko', ms=1)
        plt.colorbar(label=cb_label)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(p.with_name(f'tri_2d_{z}').with_suffix('.png'), bbox_inches='tight')
        # Interpolate
        xx = np.linspace(min(xs), max(xs))
        yy = np.linspace(min(ys), max(ys))
        X, Y = np.meshgrid(xx, yy)
        interpolators = {'linear': LinearTriInterpolator(tri, zs, trifinder=None),
                         'cubic': CubicTriInterpolator(tri, zs, kind='min_E',
                                                       trifinder=None, dz=None)}
        for k, v in interpolators.items():
            Z = v(X, Y)
            fig = go.Figure(data=[go.Surface(x=xx, y=yy, z=Z, colorscale='Jet')])
            fig.update_layout(
                autosize=True,
                # title=title.replace('\n', '<br>'),
                width=500, height=500, scene={
                    "xaxis_title": x_label,
                    "yaxis_title": y_label,
                    "zaxis_title": cb_label})
            fig.write_html(p.with_name(f'tri_3d_{k}_{z}').with_suffix('.html'))
            plt.clf()
            plt.contourf(X, Y, Z, levels=10,
                         cmap=cm.jet, alpha=1,
                         norm=plt.Normalize(vmax=Z.max(), vmin=Z.min()))
            plt.plot(xs, ys, 'ko', ms=1)
            plt.colorbar(label=cb_label)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(p.with_name(f'tri_2d_{k}_{z}').with_suffix('.png'), bbox_inches='tight')

        # Refine
        # refiner = UniformTriRefiner(tri)
        # tri_rf, zs_rf = refiner.refine_field(zs, subdiv=3)
        # fig = create_trisurf(x=tri_rf.x, y=tri_rf.y, z=zs_rf,
        #                      colormap='Jet',
        #                      show_colorbar=True,
        #                      plot_edges=False,
        #                      # title=title,
        #                      simplices=tri_rf.triangles)
        # fig.update_layout(scene={
        #         "xaxis_title": x_label,
        #         "yaxis_title": y_label,
        #         "zaxis_title": cb_label})
        # fig.write_html(p.with_name(f'tri_3d_ref_{z}').with_suffix('.html'))

        def plot_reg(xs, ys, zs, q1, q2, c='k', ls='--', label=None):
            p1, p2 = np.percentile(zs, q1), np.percentile(zs, q2)
            ids = [i for i, x in enumerate(zs) if p1 < x <= p2]
            coef = np.polyfit(xs[ids], ys[ids], 1)
            poly1d_fn = np.poly1d(coef)
            plt.plot(xs[ids], poly1d_fn(xs[ids]), color=c, linestyle=ls, label=label)

        # plot_reg(xs, ys, ts, 0, 100, c='b')
        # # plot_reg(xs, ys, 0, 25, c='g')
        # # plot_reg(xs, ys, 95, 100, c='r')
        # plot_reg(xs, ys, ts, 0, 50, c='g')
        # plot_reg(xs, ys, ys, 0, 50, c='g')

        # plot_reg(xs, ys, ts, 50, 100, c='r')
        # plot_reg(xs, ys, ts, 0, 100, c='b')
        # plot_reg(xs, ys, ts, 0, 100, c='k')

        # plot_reg(xs, ys, ts, 0, 5, c='g')
        # plot_reg(xs, ys, ys, 0, 5, c='g')

        # plot_reg(xs, ys, ys, 95, 100, c='r')
        # plot_reg(xs, ys, ts, 95, 100, c='r')

        # lgd = plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))

    p = Path(input_path)
    fixed = {'gmsh.input.geometry.rock.dz': 50.0,
             'gmsh.input.geometry.ebs.dr': 0.5}
    df = get_data(input_path, fixed)
    zs = ['fenia.output.heat.temperature.max',
          'fenia.output.heat.temperature.filling.max.value',
          'fenia.output.heat.temperature.ebs.max.value',
          'fenia.output.heat.temperature.rock.max.value',
          'fenia.output.heat.temperature.cast_iron.max.value']
    x = 'gmsh.input.geometry.borehole.dx'
    y = 'gmsh.input.geometry.ebs.dh'
    c2l = {
        'fenia.output.heat.temperature.max': 'Максимальная температура',
        'fenia.output.heat.temperature.filling.max.value': 'Максимальная температура внутри контейнера',
        'fenia.output.heat.temperature.ebs.max.value': 'Максимальная температура ИББ',
        'fenia.output.heat.temperature.rock.max.value': 'Максимальная температура вмещающей породы',
        'fenia.output.heat.temperature.cast_iron.max.value': 'Максимальная температура контейнера',
        'gmsh.input.geometry.borehole.dx': 'Расстояние между скважинами, м',
        'gmsh.input.geometry.ebs.dh': 'Расстояние между контейнерами, м'
    }
    for z in zs:
        tri_by_xyz(p, df, x, y, z,
                   x_label=c2l[x],
                   y_label=c2l[y],
                   cb_label='Температура, °C',
                   title=f'{c2l[z]} после {len(df)} расчетов'
                         f'\nпри 40 контейнерах, 10 на скважину,'
                         f'\nглубине {fixed.get("gmsh.input.geometry.rock.dz", "все")} м, '
                         f'толщине ИББ {fixed.get("gmsh.input.geometry.ebs.dr", "все")} м,'
                         f'\nтепловыделении 1000 Вт/м³')


def plot_tsne(input_path):
    # Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>

    from collections import OrderedDict
    from functools import partial
    from time import time

    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import NullFormatter

    from sklearn import manifold, datasets

    # Next line to silence pyflakes. This import is needed.
    # Axes3D
    p = Path(input_path)

    df = get_data(input_path)

    # g = sns.lmplot(data=df,
    #                x='volume.excavated',
    #                y='fenia.output.heat.temperature.max')
    # g.savefig(csv.with_name('tv').with_suffix('.png'))
    cs = ['gmsh.input.geometry.ebs.dh',
          'gmsh.input.geometry.ebs.dr',
          'gmsh.input.geometry.borehole.dx',
          'gmsh.input.geometry.rock.dz',
          'gmsh.input.geometry.rock.dnz',
          'gmsh.input.geometry.rock.dx',
          'gmsh.input.geometry.rock.dy',
          'volume.excavated',
          'fenia.output.heat.temperature.max']
    X = df[cs].to_numpy()

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X = sc.fit_transform(X)

    # n_points = 1000
    # X, color = datasets.make_s_curve(n_points, random_state=0)
    # n_neighbors = 10
    n_components = 2

    # # Create figure
    # fig = plt.figure(figsize=(15, 8))
    # fig.suptitle(
    #     "Manifold Learning with %i points, %i neighbors" % (1000, n_neighbors), fontsize=14
    # )
    #
    # # Add 3d scatter plot
    # ax = fig.add_subplot(251, projection="3d")
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    # ax.view_init(4, -72)

    # Set-up manifold methods
    # LLE = partial(
    #     manifold.LocallyLinearEmbedding,
    #     n_neighbors=n_neighbors,
    #     n_components=n_components,
    #     eigen_solver="auto",
    # )

    methods = OrderedDict()
    # methods["LLE"] = LLE(method="standard")
    # methods["LTSA"] = LLE(method="ltsa")
    # methods["Hessian LLE"] = LLE(method="hessian")
    # methods["Modified LLE"] = LLE(method="modified")
    # methods["Isomap"] = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
    methods["MDS"] = manifold.MDS(n_components, max_iter=100, n_init=1)
    # methods["SE"] = manifold.SpectralEmbedding(
    #     n_components=n_components, n_neighbors=n_neighbors
    # )
    methods["t-SNE"] = manifold.TSNE(n_components=n_components, init="pca", random_state=0)

    # Plot results
    # for i, (label, method) in enumerate(methods.items()):
    #     t0 = time()
    #     Y = method.fit_transform(X)
    #     t1 = time()
    #     print("%s: %.2g sec" % (label, t1 - t0))
    #     ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
    #     ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    #     ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
    #     ax.xaxis.set_major_formatter(NullFormatter())
    #     ax.yaxis.set_major_formatter(NullFormatter())
    #     ax.axis("tight")
    import plotly.express as px

    Y = methods['t-SNE'].fit_transform(X)

    hover = [
        'number',
        'gmsh.input.geometry.ebs.dh',
        'gmsh.input.geometry.ebs.dr',
        'gmsh.input.geometry.borehole.dx',
        'gmsh.input.geometry.rock.dz',
        'gmsh.input.geometry.rock.dnz',
        'gmsh.input.geometry.rock.dx',
        'gmsh.input.geometry.rock.dy',
        'volume.excavated',
        'fenia.output.heat.temperature.max']
    df = df[hover]
    df['x'] = Y[:, 0]
    df['y'] = Y[:, 1]
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='fenia.output.heat.temperature.max',
        # color_continuous_scale=self.results_color_scale,
        hover_name="number",
        hover_data=hover,
        # labels=labels)
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.write_html(p.with_name('t_tsne').with_suffix('.html'))

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='volume.excavated',
        # color_continuous_scale=self.results_color_scale,
        hover_name="number",
        hover_data=cs,
        # labels=labels)
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.write_html(p.with_name('ev_tsne').with_suffix('.html'))


def plot_tv(input_path):
    import plotly.express as px

    p = Path(input_path)
    df = get_data(input_path)

    # g = sns.lmplot(data=df,
    #                x='volume.excavated',
    #                y='fenia.output.heat.temperature.max')
    # g.savefig(csv.with_name('tv').with_suffix('.png'))
    cs = ['number',
          'gmsh.input.geometry.ebs.dh',
          'gmsh.input.geometry.ebs.dr',
          'gmsh.input.geometry.borehole.dx',
          'gmsh.input.geometry.rock.dz',
          'gmsh.input.geometry.rock.dnz',
          'gmsh.input.geometry.rock.dx',
          'gmsh.input.geometry.rock.dy',
          'volume.excavated',
          'fenia.output.heat.temperature.max']
    df = df[cs]
    fig = px.scatter(
        df,
        x='volume.excavated',
        y='fenia.output.heat.temperature.max',
        # color='gmsh.input.geometry.ebs.dh',
        # color_continuous_scale=self.results_color_scale,
        hover_name="number",
        hover_data=cs,
        # labels=labels
    )

    # if color is not None:
    #     fig.layout.coloraxis.colorbar.title = color
    fig.write_html(p.with_name('tv').with_suffix('.html'))


def get_data(input_path, fixed=None):
    fixed = {} if fixed is None else fixed
    p = Path(input_path)

    df = pd.read_csv(p,
                     parse_dates=['datetime_start', 'datetime_complete', 'duration'],
                     # date_parser=lambda x: pd.datetime.strptime(x, '%Y%m%d:%H:%M:%S.%f')
                     )

    df = df.rename(columns={x: x[23:] if x.startswith('user_attrs_features_~~.') else x for x in df.columns})

    # Filter
    df = df[df["state"] == "COMPLETE"]
    df = df[df["gmsh.input.geometry.container.number"] == 40]
    df = df[df["gmsh.input.geometry.container.per_borehole"] == 10]
    df = df[df["gmsh.input.geometry.ebs.dh"] <= 2.5]
    for k, v in fixed.items():
        df = df[df[k] == v]

    # Corrections
    k, t0 = 0.26, 9.0
    ts = ['fenia.output.heat.temperature.max',
          'fenia.output.heat.temperature.filling.max.value',
          'fenia.output.heat.temperature.ebs.max.value',
          'fenia.output.heat.temperature.rock.max.value',
          'fenia.output.heat.temperature.cast_iron.max.value']
    for t in ts:
        df[t] = (df[t] - t0) * k + t0
    df['gmsh.input.geometry.ebs.dh'] = df['gmsh.input.geometry.ebs.dh'] * 2

    s = 36  # Area of section of tunnel
    df['volume.excavated'] = df['fenia.output.heat.volume.filling'] \
                             + df['fenia.output.heat.volume.ebs'] \
                             + df['fenia.output.heat.volume.cast_iron'] \
                             + 3 * s * df['gmsh.input.geometry.rock.dz'] \
                             + s * df['gmsh.input.geometry.borehole.length']
    df = df[df["volume.excavated"] < 1e6]
    return df


def plot_t(input_path):
    p = Path(input_path)
    df = get_data(input_path)
    g = sns.lmplot(data=df,
                   x='volume.excavated',
                   y='fenia.output.heat.temperature.max')
    g.savefig(p.with_name('tv').with_suffix('.png'))
    df = df.melt(
        id_vars=[
            'gmsh.input.geometry.ebs.dh',
            'gmsh.input.geometry.ebs.dr',
            'gmsh.input.geometry.borehole.dx',
            'gmsh.input.geometry.rock.dz',
            'gmsh.input.geometry.rock.dnz',
            'gmsh.input.geometry.rock.dx',
            'gmsh.input.geometry.rock.dy',
            'gmsh.input.size.rock',
            'gmsh.input.size.factor',
            'fenia.output.heat.volume.filling',
            'fenia.output.heat.volume.ebs',
            'fenia.output.heat.volume.rock',
            'fenia.output.heat.volume.cast_iron',
            'volume.excavated'
        ],
        value_vars=[
            'fenia.output.heat.temperature.filling.max.value',
            'fenia.output.heat.temperature.ebs.max.value',
            'fenia.output.heat.temperature.rock.max.value',
            'fenia.output.heat.temperature.cast_iron.max.value'],
        var_name="zone",
        value_name="t")
    df = df.melt(
        id_vars=['zone', 't'],
        value_vars=[
            'gmsh.input.geometry.ebs.dh',
            'gmsh.input.geometry.ebs.dr',
            'gmsh.input.geometry.borehole.dx',
            'gmsh.input.geometry.rock.dz',
            'gmsh.input.geometry.rock.dnz',
            'gmsh.input.geometry.rock.dx',
            'gmsh.input.geometry.rock.dy',
            'gmsh.input.size.rock',
            'gmsh.input.size.factor',
            'fenia.output.heat.volume.filling',
            'fenia.output.heat.volume.ebs',
            'fenia.output.heat.volume.rock',
            'fenia.output.heat.volume.cast_iron',
            'volume.excavated'
        ],
        var_name="variable",
        value_name="value")
    print(df)
    g = sns.lmplot(data=df,
                   x='value',
                   y='t',
                   hue='zone',
                   col='variable',
                   # row=
                   # markers=['s', 'o'],
                   # palette=sns.color_palette(['r', 'g'])
                   # legend=None,
                   # order=2
                   x_jitter=.05,
                   aspect=1,
                   col_wrap=3,
                   facet_kws={'sharey': False, 'sharex': False})
    g.savefig(p.with_name('t').with_suffix('.png'))
    g = sns.displot(df, x="t", hue="zone", kind="kde", fill=True)
    g.savefig(p.with_name('t_dist').with_suffix('.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='input path')
    a = vars(parser.parse_args())  # arguments
    plot_time(**a)
    plot_tri(**a)
    plot_tsne(**a)
    plot_tv(**a)
    plot_t(**a)
