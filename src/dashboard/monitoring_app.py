import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, Any, List
import logging

try:
    from src.monitoring.metrics import MonitoringService, QueryMetrics
    from src.orchestration.workflow import OrchestrationWorkflow
    from src.prompts.registry import prompt_registry
except ImportError:
    st.error("Required modules not available. Please ensure all dependencies are installed.")
    st.stop()

logger = logging.getLogger(__name__)

class MonitoringDashboard:
    def __init__(self):
        self.monitoring_service = MonitoringService()
        self.setup_page_config()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="AI Assistant Monitoring Dashboard",
            page_icon=None,
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def display_sidebar(self):
        """Display sidebar with controls"""
        st.sidebar.title("AI Assistant Monitor")
        
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"],
            index=1
        )
        
        days_map = {
            "Last 24 hours": 1,
            "Last 7 days": 7, 
            "Last 30 days": 30,
            "All time": 90
        }
        
        self.days = days_map[time_range]
        
        if st.sidebar.button("Refresh Data"):
            st.rerun()
        
        if st.sidebar.button("Export Report"):
            self.export_report()
    
    def display_overview_metrics(self, stats: Dict[str, Any]):
        """Display overview metrics cards"""
        st.header("Overview Metrics")
        
        if "error" in stats:
            st.error(f"Error loading metrics: {stats['error']}")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Queries",
                value=f"{stats.get('total_queries', 0):,}",
                delta=None
            )
        
        with col2:
            avg_response = stats.get('avg_response_time', 0)
            st.metric(
                label="Avg Response Time",
                value=f"{avg_response:.2f}s",
                delta=None
            )
        
        with col3:
            total_tokens = stats.get('total_tokens', 0)
            st.metric(
                label="Total Tokens Used",
                value=f"{total_tokens:,}",
                delta=None
            )
        
        with col4:
            error_rate = stats.get('error_rate', 0)
            st.metric(
                label="Error Rate",
                value=f"{error_rate:.1%}",
                delta=None,
                help="Percentage of queries that resulted in errors"
            )
    
    def display_response_time_chart(self, stats: Dict[str, Any]):
        """Display response time trends"""
        st.header("Response Time Trends")
        
        if "hourly_breakdown" not in stats:
            st.warning("No hourly breakdown data available")
            return
        
        hourly_data = stats["hourly_breakdown"]
        
        if not hourly_data:
            st.warning("No hourly data available")
            return
        
        times = []
        response_times = []
        query_counts = []
        
        for time_str, data in hourly_data.items():
            times.append(datetime.fromisoformat(time_str))
            response_times.append(data.get('avg_response_time', 0))
            query_counts.append(data.get('count', 0))
        
        df = pd.DataFrame({
            'time': times,
            'response_time': response_times,
            'query_count': query_counts
        })
        
        if len(df) > 0:
            fig = px.line(
                df, 
                x='time', 
                y='response_time',
                title='Average Response Time Over Time',
                labels={'time': 'Time', 'response_time': 'Response Time (seconds)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.bar(
                df,
                x='time',
                y='query_count', 
                title='Query Volume Over Time',
                labels={'time': 'Time', 'query_count': 'Number of Queries'}
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    def display_source_distribution(self, stats: Dict[str, Any]):
        """Display query source distribution"""
        st.header("Query Source Distribution")
        
        queries_by_source = stats.get('queries_by_source', {})
        
        if not queries_by_source:
            st.warning("No source distribution data available")
            return
        
        labels = list(queries_by_source.keys())
        values = list(queries_by_source.values())
        
        fig = px.pie(
            values=values,
            names=labels,
            title='Distribution of Queries by Source'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        df = pd.DataFrame({
            'Source': labels,
            'Count': values,
            'Percentage': [f"{v/sum(values)*100:.1f}%" for v in values]
        })
        st.dataframe(df, use_container_width=True)
    
    def display_quality_metrics(self, stats: Dict[str, Any]):
        """Display quality metrics"""
        st.header("Quality Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_confidence = stats.get('avg_retrieval_confidence', 0)
            if avg_confidence is not None:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = avg_confidence,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Avg Retrieval Confidence"},
                    gauge = {'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.8], 'color': "yellow"},
                                {'range': [0.8, 1], 'color': "green"}],
                            'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 0.9}}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_rating = stats.get('avg_user_rating', 0)
            if avg_rating is not None:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = avg_rating,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Avg User Rating"},
                    gauge = {'axis': {'range': [0, 5]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, 2], 'color': "lightgray"},
                                {'range': [2, 4], 'color': "yellow"},
                                {'range': [4, 5], 'color': "green"}],
                            'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 4.5}}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    def display_prompt_performance(self):
        """Display prompt performance metrics"""
        st.header("Prompt Performance")
        
        try:
            prompts = prompt_registry.list_prompts()
            
            if not prompts:
                st.warning("No prompts available")
                return
            
            prompt_data = []
            for prompt in prompts:
                stats = prompt_registry.get_performance_stats(prompt["name"])
                prompt_data.append({
                    'Name': prompt['name'],
                    'Version': prompt['version'],
                    'Description': prompt['description'],
                    'Accuracy': f"{stats['accuracy']:.2%}",
                    'Avg Tokens': f"{stats['avg_tokens']:.0f}",
                    'Avg Response Time': f"{stats['avg_response_time']:.2f}s",
                    'Total Requests': stats['total_requests']
                })
            
            df = pd.DataFrame(prompt_data)
            st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading prompt performance: {e}")
    
    def display_evidently_report(self):
        """Display Evidently AI report"""
        st.header("Data Drift Report")
        
        try:
            report = self.monitoring_service.generate_evidently_report(days=self.days)
            
            if report is None:
                st.warning("No drift report available. Insufficient historical data.")
                return
            
            html_report = report.get_html()
            st.components.v1.html(html_report, height=800, scrolling=True)
            
        except Exception as e:
            st.error(f"Error generating drift report: {e}")
    
    def export_report(self):
        """Export monitoring report"""
        try:
            stats = self.monitoring_service.get_performance_statistics(days=self.days)
            
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "time_range_days": self.days,
                "statistics": stats,
                "prompts": prompt_registry.list_prompts()
            }
            
            report_json = json.dumps(report_data, indent=2, default=str)
            
            st.download_button(
                label="Download Report",
                data=report_json,
                file_name=f"ai_assistant_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Error exporting report: {e}")
    
    def run(self):
        """Run the monitoring dashboard"""
        st.title("AI Assistant Monitoring Dashboard")
        
        self.display_sidebar()
        
        with st.spinner("Loading monitoring data..."):
            stats = self.monitoring_service.create_monitoring_dashboard_data()
        
        self.display_overview_metrics(stats)
        
        st.divider()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Performance", 
            "Source Distribution", 
            "Quality", 
            "Prompts", 
            "Data Drift"
        ])
        
        with tab1:
            self.display_response_time_chart(stats)
        
        with tab2:
            self.display_source_distribution(stats)
        
        with tab3:
            self.display_quality_metrics(stats)
        
        with tab4:
            self.display_prompt_performance()
        
        with tab5:
            self.display_evidently_report()
        
        st.divider()
        st.markdown(
            """
            <div style='text-align: center; color: gray;'>
                AI Assistant Monitoring Dashboard - Powered by Evidently AI & Streamlit
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    """Main function to run the dashboard"""
    dashboard = MonitoringDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
