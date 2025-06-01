from BFTUSk.utils import *

def get_result_df(filename):
    
    df = pd.read_csv(filename)

    for endswith in [
        ".keys", 
        ".rt", 
        ".duration"]:
        df = df.drop([c for c in df.columns if c.endswith(endswith)], axis="columns")

    # drop routines information
    df = df.drop([c for c in df.columns if c.endswith(".started") or c.endswith(".stopped")], axis="columns")
    # drop loop information
    df = df.drop([c for c in df.columns if c.startswith("trials.") or c.startswith("breaks.")], axis="columns")
    # drop other unnecessary columns
    df = df.drop(["thisRow.t", "notes"], axis="columns")
    df = df.drop([c for c in df.columns if c.startswith("Unnamed")], axis="columns")
    # drop non-trials
    df = df.dropna(subset=['rt_ms', 'reward'], how='all')

    df["reward_chance_endpoint"] = [np.round(val, 2) for val in df["reward_chance_endpoint"]]

    for list_var in [
        "likelihoods", 
        "bubble_size_steps_no_noise", 
        "reward_chances"
        ]:
        if list_var in df.columns:
            df[list_var] = [ast.literal_eval(e) for e in df[list_var]]

    df["cum_reward"] = np.cumsum([val*won for val, won in zip(df["reward_size"], df["reward"])])

    df = df.reset_index()

    return df

def plot_cum_reward(df):
    plt.figure(figsize=(4,2))
    plt.plot(df["cum_reward"])
    plt.hlines(5, 0, len(df), colors="red")
    plt.xlabel("trial N"); plt.ylabel("gain (€)"); plt.title("cumulative reward")
    plt.tight_layout()
    plt.show()

def plot_avg_per_cond(df):
    f, axes = plt.subplots(1, 3, figsize=(9,3))
    for ax, var, title in zip(axes, ["reward_chance_endpoint", "reward_size", "reward"], ["reward endpoints", "reward size (€)", "reward"]):
        # reward_chance_endpoint
        ax.set_title(title)
        var_steps = sorted(list(set(df[var])))
        ax.pie(
            [len(df[df[var] == val]) for val in var_steps], 
            labels = var_steps,
            autopct='%i%%'
            )
    plt.tight_layout()
    plt.show()

def plot_endpoint_variations(df):
    plt.figure(figsize=(3,5))
    bubble_inds = sorted(list(set(df["sample_ind"])))
    endpoints = sorted(list(set(df["reward_chance_endpoint"])))
    cmap = plt.get_cmap('viridis')
    colors = dict((e, c) for e, c in zip(endpoints, [cmap(i / (len(endpoints) - 1)) for i in range(len(endpoints))]))
    for bubble_ind in bubble_inds:
        rows = df[df["sample_ind"] == bubble_ind]
        plt.scatter(
            rows["sample_ind"], [e*100 for e in rows["likelihood"]], 
            color=[colors[e] for e in rows["reward_chance_endpoint"]],
            marker="x")
    plt.title(f"SNR: {rows['snr_db'].tolist()[0]:.0f} dB")
    plt.xlabel("bubble ind"); plt.ylabel("bubble size\n(screen height %)")
    # plt.legend(colors, title="test", ncols=2)
    cbar = plt.colorbar()
    cbar.set_label('endpoint reward prob.')
    plt.show()

def plot_avg_reward_prob_per_bubble_step(df):
    plt.figure(figsize=(2,3.5))
    bubble_inds = sorted(list(set(df["sample_ind"])))
    endpoints = sorted(list(set(df["reward_chance_endpoint"])))
    cmap = plt.get_cmap('viridis')
    colors = dict((e, c) for e, c in zip(endpoints, [cmap(i / (len(endpoints) - 1)) for i in range(len(endpoints))]))
    for endpoint in endpoints[-1::-1]:
        plt.plot(
            bubble_inds, [
                np.mean(df[
                    (df["sample_ind"] == bubble_ind) & 
                    (df["reward_chance_endpoint"] == endpoint)
                ]["reward"]) for bubble_ind in bubble_inds
            ], color=colors[endpoint], label=endpoint, linestyle='', marker="x")
    plt.xlabel("bubble ind"); plt.ylabel("reward prob.")
    plt.legend(title="endpoint reward prob.", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()