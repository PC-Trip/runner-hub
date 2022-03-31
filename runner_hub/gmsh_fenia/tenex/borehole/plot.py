from pathlib import Path

import pandas as pd
import seaborn as sns


def plot_tsne():
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

    root = Path()
    csv = root / "dag/work/tenex_borehole_1/data.csv"
    df = pd.read_csv(csv)
    print(df)

    df = df.rename(columns={x: x[23:] if x.startswith('user_attrs_features_~~.') else x for x in df.columns})
    df = df[df["state"] == "COMPLETE"]
    df = df[df["gmsh.input.geometry.container.number"] == 40]
    df = df[df["gmsh.input.geometry.container.per_borehole"] == 10]
    s = 36
    df['volume.excavated'] = df['fenia.output.heat.volume.filling'] \
                             + df['fenia.output.heat.volume.ebs'] \
                             + df['fenia.output.heat.volume.cast_iron'] \
                             + 3 * s * df['gmsh.input.geometry.rock.dz'] \
                             + s * df['gmsh.input.geometry.borehole.length']
    df = df[df["volume.excavated"] < 1e6]

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

    X = df.to_numpy()

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

    df['x'] = Y[:, 0]
    df['y'] = Y[:, 1]
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='fenia.output.heat.temperature.max',
        # color_continuous_scale=self.results_color_scale,
        hover_name="number",
        hover_data=cs,
        # labels=labels)
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.write_html(csv.with_name('t_tsne').with_suffix('.html'))

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
    fig.write_html(csv.with_name('ev_tsne').with_suffix('.html'))


def plot_tv():
    import plotly.express as px

    root = Path()
    csv = root / "dag/work/tenex_borehole_1/data.csv"
    df = pd.read_csv(csv)
    print(df)

    df = df.rename(columns={x: x[23:] if x.startswith('user_attrs_features_~~.') else x for x in df.columns})
    df = df[df["state"] == "COMPLETE"]
    df = df[df["gmsh.input.geometry.container.number"] == 40]
    df = df[df["gmsh.input.geometry.container.per_borehole"] == 10]
    s = 36
    df['volume.excavated'] = df['fenia.output.heat.volume.filling'] \
                             + df['fenia.output.heat.volume.ebs'] \
                             + df['fenia.output.heat.volume.cast_iron'] \
                             + 3 * s * df['gmsh.input.geometry.rock.dz'] \
                             + s * df['gmsh.input.geometry.borehole.length']
    df = df[df["volume.excavated"] < 1e6]

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
    fig.write_html(csv.with_name('tv').with_suffix('.html'))


def plot_t():
    root = Path()
    csv = root / "dag/work/tenex_borehole_1/data.csv"
    df = pd.read_csv(csv)
    print(df)
    df = df.rename(columns={x: x[23:] if x.startswith('user_attrs_features_~~.') else x for x in df.columns})
    df = df[df["state"] == "COMPLETE"]
    df = df[df["gmsh.input.geometry.container.number"] == 40]
    df = df[df["gmsh.input.geometry.container.per_borehole"] == 10]
    s = 36
    df['volume.excavated'] = df['fenia.output.heat.volume.filling'] \
                             + df['fenia.output.heat.volume.ebs'] \
                             + df['fenia.output.heat.volume.cast_iron']
    # + 3 * s * df['gmsh.input.geometry.rock.dz'] \
    # + s * df['gmsh.input.geometry.borehole.length']
    df = df[df["volume.excavated"] < 1e6]
    g = sns.lmplot(data=df,
                   x='volume.excavated',
                   y='fenia.output.heat.temperature.max')
    g.savefig(csv.with_name('tv').with_suffix('.png'))

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
    g.savefig(csv.with_name('t').with_suffix('.png'))
    g = sns.displot(df, x="t", hue="zone", kind="kde", fill=True)
    g.savefig(csv.with_name('t_dist').with_suffix('.png'))


if __name__ == '__main__':
    plot_tsne()
    plot_tv()
    plot_t()
