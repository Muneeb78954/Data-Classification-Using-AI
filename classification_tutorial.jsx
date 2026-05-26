import { useState, useEffect } from "react";

// ── Iris-like dataset (150 samples, 4 features, 3 classes) ──────────────────
function generateDataset() {
  const rng = (mu, sigma) => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const samples = [];
  const specs = [
    { label: "Setosa",     id: 0, sl:[5.0,0.35], sw:[3.4,0.38], pl:[1.5,0.17], pw:[0.25,0.11] },
    { label: "Versicolor", id: 1, sl:[5.9,0.51], sw:[2.8,0.31], pl:[4.3,0.47], pw:[1.33,0.20] },
    { label: "Virginica",  id: 2, sl:[6.6,0.63], sw:[3.0,0.32], pl:[5.6,0.55], pw:[2.03,0.27] },
  ];
  specs.forEach(s => {
    for (let i = 0; i < 50; i++) {
      samples.push({
        sepalLength: +rng(...s.sl).toFixed(1),
        sepalWidth:  +rng(...s.sw).toFixed(1),
        petalLength: +rng(...s.pl).toFixed(1),
        petalWidth:  +rng(...s.pw).toFixed(1),
        label: s.label, classId: s.id,
      });
    }
  });
  return samples.sort(() => Math.random() - 0.5);
}

// ── Train/Test split ────────────────────────────────────────────────────────
function trainTestSplit(data, testRatio = 0.2) {
  const n = Math.round(data.length * testRatio);
  return { train: data.slice(n), test: data.slice(0, n) };
}

// ── KNN classifier ──────────────────────────────────────────────────────────
function euclidean(a, b) {
  const keys = ["sepalLength","sepalWidth","petalLength","petalWidth"];
  return Math.sqrt(keys.reduce((s, k) => s + (a[k]-b[k])**2, 0));
}
function knnPredict(train, point, k = 5) {
  const dists = train.map(t => ({ ...t, dist: euclidean(t, point) }))
                     .sort((a, b) => a.dist - b.dist).slice(0, k);
  const votes = {};
  dists.forEach(d => { votes[d.label] = (votes[d.label] || 0) + 1; });
  return Object.entries(votes).sort((a,b) => b[1]-a[1])[0][0];
}
function evaluate(train, test, k) {
  let correct = 0;
  const results = test.map(t => {
    const pred = knnPredict(train, t, k);
    if (pred === t.label) correct++;
    return { ...t, predicted: pred, correct: pred === t.label };
  });
  return { accuracy: correct / test.length, results };
}

// ── Confusion matrix ────────────────────────────────────────────────────────
function confusionMatrix(results, classes) {
  const m = {};
  classes.forEach(a => { m[a] = {}; classes.forEach(b => { m[a][b] = 0; }); });
  results.forEach(r => m[r.label][r.predicted]++);
  return m;
}

// ── Colour map ──────────────────────────────────────────────────────────────
const COLOR = { Setosa: "#4ade80", Versicolor: "#60a5fa", Virginica: "#f472b6" };
const CLASSES = ["Setosa", "Versicolor", "Virginica"];

// ═══════════════════════════════════════════════════════════════════════════
export default function App() {
  const [dataset] = useState(() => generateDataset());
  const [testRatio, setTestRatio] = useState(0.2);
  const [k, setK]                 = useState(5);
  const [split, setSplit]         = useState(null);
  const [evalResult, setEvalResult] = useState(null);
  const [step, setStep]           = useState(0); // 0=load 1=split 2=train 3=eval
  const [animating, setAnimating] = useState(false);

  const advance = () => {
    if (animating) return;
    setAnimating(true);
    setTimeout(() => {
      if (step === 0) {
        setStep(1);
      } else if (step === 1) {
        setSplit(trainTestSplit(dataset, testRatio));
        setStep(2);
      } else if (step === 2) {
        const sp = split || trainTestSplit(dataset, testRatio);
        setSplit(sp);
        const res = evaluate(sp.train, sp.test, k);
        setEvalResult(res);
        setStep(3);
      } else if (step === 3) {
        setStep(4);
      }
      setAnimating(false);
    }, 350);
  };

  const reset = () => { setStep(0); setSplit(null); setEvalResult(null); };

  const cm = evalResult ? confusionMatrix(evalResult.results, CLASSES) : null;

  const stepLabels = ["Load Dataset","Split Data","Train Model","Evaluate","Results ✓"];

  return (
    <div style={{
      minHeight:"100vh", background:"#0a0a0f",
      fontFamily:"'DM Mono', 'Fira Code', monospace",
      color:"#e2e8f0", padding:"0 0 60px",
    }}>
      {/* ── Header ── */}
      <div style={{
        background:"linear-gradient(135deg,#1e1b4b,#0f172a)",
        borderBottom:"1px solid #312e81",
        padding:"32px 40px 24px",
      }}>
        <div style={{maxWidth:900,margin:"0 auto"}}>
          <div style={{fontSize:11,letterSpacing:4,color:"#818cf8",marginBottom:8,textTransform:"uppercase"}}>
            Machine Learning · Classification
          </div>
          <h1 style={{fontSize:28,fontWeight:700,margin:0,
            background:"linear-gradient(90deg,#a5b4fc,#f0abfc)",
            WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
            K-Nearest Neighbours on Iris
          </h1>
          <p style={{margin:"10px 0 0",color:"#94a3b8",fontSize:13,lineHeight:1.6}}>
            Step through data loading → splitting → training → evaluation interactively.
          </p>
        </div>
      </div>

      <div style={{maxWidth:900,margin:"0 auto",padding:"32px 20px"}}>

        {/* ── Stepper ── */}
        <div style={{display:"flex",gap:0,marginBottom:36,overflowX:"auto"}}>
          {stepLabels.map((l,i) => (
            <div key={i} style={{display:"flex",alignItems:"center",flex:i<4?1:0}}>
              <div style={{
                display:"flex",flexDirection:"column",alignItems:"center",gap:4,
                opacity: i <= step ? 1 : 0.35, transition:"opacity .4s",
              }}>
                <div style={{
                  width:32,height:32,borderRadius:"50%",display:"flex",
                  alignItems:"center",justifyContent:"center",fontSize:13,fontWeight:700,
                  background: i < step ? "#4ade80" : i===step ? "#818cf8" : "#1e293b",
                  color: i <= step ? "#0a0a0f" : "#64748b",
                  border: i===step ? "2px solid #a5b4fc" : "2px solid transparent",
                  transition:"all .4s",
                }}>
                  {i < step ? "✓" : i+1}
                </div>
                <span style={{fontSize:10,letterSpacing:1,color:i===step?"#a5b4fc":"#64748b",whiteSpace:"nowrap"}}>
                  {l}
                </span>
              </div>
              {i < 4 && <div style={{flex:1,height:2,margin:"0 6px",marginBottom:18,
                background: i < step ? "#4ade80" : "#1e293b",transition:"background .4s"}}/>}
            </div>
          ))}
        </div>

        {/* ── STEP 0: Dataset overview ── */}
        {step >= 0 && (
          <Card title="📦 Step 1 — Load Dataset" accent="#818cf8">
            <p style={{color:"#94a3b8",fontSize:13,marginTop:0}}>
              We use a synthetic Iris-like dataset with <b style={{color:"#e2e8f0"}}>150 samples</b>,&nbsp;
              <b style={{color:"#e2e8f0"}}>4 numerical features</b>, and&nbsp;
              <b style={{color:"#e2e8f0"}}>3 flower classes</b>.
            </p>
            {/* Feature table header */}
            <div style={{overflowX:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
                <thead>
                  <tr style={{borderBottom:"1px solid #1e293b"}}>
                    {["sepalLength","sepalWidth","petalLength","petalWidth","label"].map(h => (
                      <th key={h} style={{padding:"6px 10px",textAlign:"left",
                        color:"#818cf8",fontWeight:600,letterSpacing:1,fontSize:10}}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataset.slice(0,6).map((r,i) => (
                    <tr key={i} style={{borderBottom:"1px solid #0f172a"}}>
                      {["sepalLength","sepalWidth","petalLength","petalWidth"].map(k => (
                        <td key={k} style={{padding:"5px 10px",color:"#cbd5e1"}}>{r[k]}</td>
                      ))}
                      <td style={{padding:"5px 10px"}}>
                        <Badge color={COLOR[r.label]}>{r.label}</Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{marginTop:12,fontSize:12,color:"#475569"}}>
              Showing 6 of 150 rows · shuffled randomly
            </div>
            {/* Class distribution */}
            <div style={{marginTop:16,display:"flex",gap:12,flexWrap:"wrap"}}>
              {CLASSES.map(c => (
                <div key={c} style={{
                  padding:"6px 14px",borderRadius:8,fontSize:12,
                  background:COLOR[c]+"22",border:`1px solid ${COLOR[c]}55`,color:COLOR[c],
                }}>
                  {c}: 50 samples
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* ── STEP 1: Split ── */}
        {step >= 1 && (
          <Card title="✂️  Step 2 — Train / Test Split" accent="#60a5fa" delay>
            <p style={{color:"#94a3b8",fontSize:13,marginTop:0}}>
              Separate labelled data so the model trains on one portion and is tested on unseen samples.
            </p>
            <label style={{fontSize:12,color:"#94a3b8"}}>
              Test ratio: <b style={{color:"#60a5fa"}}>{(testRatio*100).toFixed(0)}%</b>
            </label>
            <input type="range" min={10} max={40} step={5}
              value={testRatio*100}
              onChange={e => { setTestRatio(e.target.value/100); setSplit(null); setEvalResult(null); if(step>2)setStep(2); }}
              style={{width:"100%",accentColor:"#60a5fa",margin:"6px 0 14px"}}
            />
            <div style={{display:"flex",gap:12,flexWrap:"wrap"}}>
              <StatBox label="Training samples" value={Math.round(150*(1-testRatio))} color="#4ade80"/>
              <StatBox label="Test samples"     value={Math.round(150*testRatio)}     color="#f472b6"/>
              <StatBox label="Features"         value={4}                              color="#60a5fa"/>
              <StatBox label="Classes"          value={3}                              color="#fbbf24"/>
            </div>
            <CodeBlock>{`from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=${testRatio}, random_state=42
)
# Train: ${Math.round(150*(1-testRatio))} samples  |  Test: ${Math.round(150*testRatio)} samples`}</CodeBlock>
          </Card>
        )}

        {/* ── STEP 2: Train ── */}
        {step >= 2 && (
          <Card title="🤖  Step 3 — Train the Model (KNN)" accent="#4ade80" delay>
            <p style={{color:"#94a3b8",fontSize:13,marginTop:0}}>
              K-Nearest Neighbours classifies a sample by majority vote among its <em>k</em> closest training points.
              No explicit "training" phase — the algorithm memorises the training set.
            </p>
            <label style={{fontSize:12,color:"#94a3b8"}}>
              k (neighbours): <b style={{color:"#4ade80"}}>{k}</b>
            </label>
            <input type="range" min={1} max={15} step={2}
              value={k}
              onChange={e => { setK(+e.target.value); setEvalResult(null); if(step>3)setStep(3); }}
              style={{width:"100%",accentColor:"#4ade80",margin:"6px 0 14px"}}
            />
            <CodeBlock>{`from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=${k})
model.fit(X_train, y_train)          # store training data
# Prediction: find ${k} nearest neighbours → majority vote`}</CodeBlock>
            <div style={{marginTop:14,padding:12,borderRadius:8,
              background:"#0f172a",border:"1px solid #1e293b",fontSize:12,color:"#94a3b8"}}>
              <span style={{color:"#4ade80",fontWeight:600}}>How it works:</span> For each test point,
              compute Euclidean distance to every training point, pick the {k} smallest,
              return the most common class label.
            </div>
          </Card>
        )}

        {/* ── STEP 3: Evaluate ── */}
        {step >= 3 && evalResult && (
          <Card title="📊  Step 4 — Evaluate" accent="#f472b6" delay>
            <div style={{display:"flex",gap:16,flexWrap:"wrap",marginBottom:20}}>
              <BigStat label="Accuracy" value={(evalResult.accuracy*100).toFixed(1)+"%"}
                color={evalResult.accuracy>0.9?"#4ade80":evalResult.accuracy>0.7?"#fbbf24":"#f87171"}/>
              <BigStat label="Test size" value={evalResult.results.length} color="#60a5fa"/>
              <BigStat label="Correct"   value={evalResult.results.filter(r=>r.correct).length} color="#4ade80"/>
              <BigStat label="Wrong"     value={evalResult.results.filter(r=>!r.correct).length} color="#f87171"/>
            </div>

            {/* Confusion matrix */}
            <div style={{fontSize:12,color:"#818cf8",marginBottom:8,fontWeight:600,letterSpacing:1}}>
              CONFUSION MATRIX
            </div>
            <div style={{overflowX:"auto"}}>
              <table style={{borderCollapse:"collapse",fontSize:12}}>
                <thead>
                  <tr>
                    <th style={{padding:"4px 12px",color:"#475569",textAlign:"left"}}>Actual ↓ / Predicted →</th>
                    {CLASSES.map(c => (
                      <th key={c} style={{padding:"4px 12px",color:COLOR[c]}}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {CLASSES.map(actual => (
                    <tr key={actual}>
                      <td style={{padding:"4px 12px",color:COLOR[actual],fontWeight:600}}>{actual}</td>
                      {CLASSES.map(pred => {
                        const val = cm[actual][pred];
                        const isCorrect = actual===pred;
                        return (
                          <td key={pred} style={{
                            padding:"4px 12px",textAlign:"center",
                            background: isCorrect
                              ? (val > 0 ? COLOR[actual]+"33" : "#0f172a")
                              : (val > 0 ? "#f8717133" : "#0f172a"),
                            color: isCorrect ? COLOR[actual] : val>0?"#f87171":"#334155",
                            borderRadius:4,fontWeight: isCorrect?"700":"400",
                          }}>
                            {val}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Per-sample strip */}
            <div style={{marginTop:20,fontSize:12,color:"#818cf8",fontWeight:600,letterSpacing:1,marginBottom:8}}>
              TEST PREDICTIONS (first 30)
            </div>
            <div style={{display:"flex",flexWrap:"wrap",gap:4}}>
              {evalResult.results.slice(0,30).map((r,i) => (
                <div key={i} title={`${r.label} → ${r.predicted}`}
                  style={{
                    width:22,height:22,borderRadius:4,display:"flex",
                    alignItems:"center",justifyContent:"center",fontSize:10,
                    background: r.correct ? COLOR[r.label]+"44" : "#f8717133",
                    border:`1px solid ${r.correct ? COLOR[r.label] : "#f87171"}`,
                    color: r.correct ? COLOR[r.label] : "#f87171",
                    cursor:"default",
                  }}>
                  {r.correct ? "✓" : "✗"}
                </div>
              ))}
            </div>

            <CodeBlock>{`y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {(evalResult.accuracy*100).toFixed(1)}%")
print(classification_report(y_test, y_pred))`}</CodeBlock>
          </Card>
        )}

        {/* ── Controls ── */}
        <div style={{display:"flex",gap:12,marginTop:28,flexWrap:"wrap"}}>
          {step < 4 && (
            <button onClick={advance} disabled={animating} style={{
              padding:"12px 28px",borderRadius:10,border:"none",cursor:"pointer",
              background:"linear-gradient(135deg,#4f46e5,#7c3aed)",
              color:"#fff",fontSize:14,fontWeight:700,letterSpacing:0.5,
              opacity:animating?0.6:1,transition:"opacity .2s",
              boxShadow:"0 4px 20px #4f46e544",
            }}>
              {step===0?"Load Dataset →":step===1?"Split Data →":step===2?"Train Model →":"Evaluate →"}
            </button>
          )}
          {step > 0 && (
            <button onClick={reset} style={{
              padding:"12px 20px",borderRadius:10,cursor:"pointer",
              background:"transparent",border:"1px solid #334155",
              color:"#64748b",fontSize:14,fontWeight:600,
            }}>
              ↺ Reset
            </button>
          )}
        </div>

        {/* ── Concept glossary ── */}
        <div style={{marginTop:48,padding:24,borderRadius:14,
          background:"#0f172a",border:"1px solid #1e293b"}}>
          <div style={{fontSize:11,letterSpacing:3,color:"#475569",marginBottom:16,textTransform:"uppercase"}}>
            Key Concepts
          </div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(240px,1fr))",gap:16}}>
            {[
              {t:"Features (X)",d:"Input variables the model uses to make predictions.",c:"#818cf8"},
              {t:"Labels (y)",  d:"The ground-truth class each sample belongs to.",c:"#4ade80"},
              {t:"Train set",   d:"Data the model learns from during training.",c:"#60a5fa"},
              {t:"Test set",    d:"Held-out data used to measure generalisation.",c:"#f472b6"},
              {t:"KNN",         d:"Predicts by majority vote among k nearest neighbours.",c:"#fbbf24"},
              {t:"Accuracy",    d:"Fraction of test samples correctly classified.",c:"#f97316"},
            ].map(({t,d,c}) => (
              <div key={t}>
                <div style={{fontSize:12,fontWeight:700,color:c,marginBottom:4}}>{t}</div>
                <div style={{fontSize:12,color:"#64748b",lineHeight:1.5}}>{d}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────
function Card({ title, accent, children, delay }) {
  return (
    <div style={{
      marginBottom:20,padding:24,borderRadius:14,
      background:"#0f172a",border:`1px solid ${accent}33`,
      animation: delay ? "fadeUp .4s ease" : "none",
    }}>
      <style>{`@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}`}</style>
      <div style={{fontSize:14,fontWeight:700,color:accent,marginBottom:14}}>{title}</div>
      {children}
    </div>
  );
}

function Badge({ color, children }) {
  return (
    <span style={{
      padding:"2px 8px",borderRadius:999,fontSize:11,fontWeight:600,
      background:color+"22",color,border:`1px solid ${color}55`,
    }}>{children}</span>
  );
}

function StatBox({ label, value, color }) {
  return (
    <div style={{
      padding:"10px 16px",borderRadius:10,minWidth:90,
      background:color+"11",border:`1px solid ${color}33`,
    }}>
      <div style={{fontSize:22,fontWeight:800,color}}>{value}</div>
      <div style={{fontSize:11,color:"#64748b",marginTop:2}}>{label}</div>
    </div>
  );
}

function BigStat({ label, value, color }) {
  return (
    <div style={{
      flex:1,minWidth:100,padding:"14px 18px",borderRadius:12,
      background:color+"11",border:`1px solid ${color}33`,textAlign:"center",
    }}>
      <div style={{fontSize:26,fontWeight:800,color}}>{value}</div>
      <div style={{fontSize:11,color:"#94a3b8",marginTop:2}}>{label}</div>
    </div>
  );
}

function CodeBlock({ children }) {
  return (
    <pre style={{
      margin:"14px 0 0",padding:"14px 16px",borderRadius:10,overflowX:"auto",
      background:"#020617",border:"1px solid #1e293b",
      fontSize:11,lineHeight:1.7,color:"#94a3b8",whiteSpace:"pre-wrap",
    }}>{children}</pre>
  );
}
