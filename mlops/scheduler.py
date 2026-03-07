"""
APScheduler — runs pipeline daily at market open
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False

from mlops.train_pipeline import TrainingPipeline
from datetime import datetime

def run_daily_pipeline():
    print(f"\n[Scheduler] Triggered at {datetime.now()}")
    pipeline = TrainingPipeline()
    result   = pipeline.run(
        tickers=["SPY","AAPL","MSFT","NVDA","GOOGL","JPM","JNJ","XOM"],
        period="1y"
    )
    print(f"[Scheduler] Done. Runs: {result['runs']} Best Sharpe: {result['best_sharpe']:.4f}")

if __name__ == "__main__":
    # Run once immediately
    run_daily_pipeline()

    if HAS_SCHEDULER:
        scheduler = BlockingScheduler()
        # Every weekday at 9:35 AM ET (market open + 5min)
        scheduler.add_job(run_daily_pipeline, CronTrigger(
            day_of_week="mon-fri", hour=9, minute=35,
            timezone="America/New_York"))
        print("\n[Scheduler] Running. Next trigger: weekdays 9:35 AM ET")
        print("Press Ctrl+C to stop")
        try:
            scheduler.start()
        except KeyboardInterrupt:
            print("[Scheduler] Stopped")
    else:
        print("[Scheduler] Install apscheduler for daily scheduling:")
        print("  pip install apscheduler")
