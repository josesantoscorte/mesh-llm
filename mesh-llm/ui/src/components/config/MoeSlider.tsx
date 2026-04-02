import type { AggregatedModelMoe } from '../../lib/models';

type MoeSliderProps = {
  modelName: string;
  moe: AggregatedModelMoe;
  currentExperts: number;
  modelSizeBytes: number;
  onExpertsChange: (n: number) => void;
};

function formatVramImpact(experts: number, totalExperts: number, modelSizeBytes: number) {
  const gb = (experts * modelSizeBytes) / totalExperts / 1e9;
  return `${gb.toFixed(1)} GB for ${experts} expert${experts !== 1 ? 's' : ''}`;
}

export function MoeSlider({ modelName, moe, currentExperts, modelSizeBytes, onExpertsChange }: MoeSliderProps) {
  const value = Math.max(1, Math.min(currentExperts, moe.nExpert));

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={`moe-slider-${modelName}`} className="text-xs font-medium text-foreground">
          Experts
        </label>
        <span className="tabular-nums text-xs text-muted-foreground font-mono">
          {value} of {moe.nExpert}
        </span>
      </div>
      <input
        id={`moe-slider-${modelName}`}
        data-testid="moe-slider"
        type="range"
        min={1}
        max={moe.nExpert}
        step={1}
        value={value}
        onChange={(e) => onExpertsChange(parseInt(e.target.value, 10))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-muted/60 accent-primary focus-visible:outline-none
          [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:rounded-full
          [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-primary
          [&::-moz-range-thumb]:transition-transform [&::-moz-range-thumb]:duration-150 [&::-moz-range-thumb]:hover:scale-125
          [&::-moz-range-track]:h-1.5 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-muted/60
          [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary
          [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:duration-150 [&::-webkit-slider-thumb]:hover:scale-125
          [&:focus-visible::-webkit-slider-thumb]:shadow-[0_0_0_3px_hsl(var(--ring))]
          [&:focus-visible::-moz-range-thumb]:shadow-[0_0_0_3px_hsl(var(--ring))]"
      />
      <div className="text-xs text-muted-foreground font-mono">
        {formatVramImpact(value, moe.nExpert, modelSizeBytes)}
      </div>
    </div>
  );
}
