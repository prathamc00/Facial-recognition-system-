import os
import csv
from datetime import datetime
import pandas as pd


class AttendanceLogger:
    def __init__(self, log_dir='../logs'):
        """
        Initialize the attendance logger.
        
        Args:
            log_dir (str): Directory to save attendance logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Current session attendance file
        today = datetime.now().strftime('%Y-%m-%d')
        self.attendance_file = os.path.join(log_dir, f'attendance_{today}.csv')
        
        # Initialize file if it doesn't exist
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Name', 'Confidence', 'Status'])
        
        # Track recent recognitions to avoid duplicates
        self.recent_recognitions = {}  # {name: timestamp}
        self.duplicate_threshold = 30  # seconds
    
    def log_attendance(self, name, confidence, status='Present'):
        """
        Log an attendance entry.
        
        Args:
            name (str): Person's name
            confidence (float): Recognition confidence (0-1)
            status (str): Status (e.g., 'Present', 'Late', 'Left')
        
        Returns:
            bool: True if logged, False if duplicate within threshold
        """
        current_time = datetime.now()
        
        # Check for duplicate entries
        if name in self.recent_recognitions:
            time_diff = (current_time - self.recent_recognitions[name]).total_seconds()
            if time_diff < self.duplicate_threshold:
                return False  # Skip duplicate
        
        # Log the attendance
        timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, name, f'{confidence:.2f}', status])
        
        # Update recent recognitions
        self.recent_recognitions[name] = current_time
        
        return True
    
    def get_todays_attendance(self):
        """
        Get today's attendance records.
        
        Returns:
            list: List of attendance records as dictionaries
        """
        if not os.path.exists(self.attendance_file):
            return []
        
        attendance = []
        with open(self.attendance_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                attendance.append(row)
        
        return attendance
    
    def get_unique_attendees_today(self):
        """
        Get list of unique people who attended today.
        
        Returns:
            list: List of unique names
        """
        attendance = self.get_todays_attendance()
        unique_names = set()
        
        for record in attendance:
            unique_names.add(record['Name'])
        
        return sorted(list(unique_names))
    
    def export_to_excel(self, start_date=None, end_date=None, output_file=None):
        """
        Export attendance logs to Excel file.
        
        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            output_file (str, optional): Output filename
        
        Returns:
            str: Path to the exported file
        """
        try:
            # Find all attendance CSV files
            attendance_files = []
            for filename in os.listdir(self.log_dir):
                if filename.startswith('attendance_') and filename.endswith('.csv'):
                    attendance_files.append(os.path.join(self.log_dir, filename))
            
            if not attendance_files:
                return None
            
            # Read all files into a single DataFrame
            dfs = []
            for file in attendance_files:
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            
            if not dfs:
                return None
            
            # Combine all DataFrames
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Convert Timestamp column to datetime
            combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
            
            # Filter by date range if specified
            if start_date:
                combined_df = combined_df[combined_df['Timestamp'] >= pd.to_datetime(start_date)]
            if end_date:
                combined_df = combined_df[combined_df['Timestamp'] <= pd.to_datetime(end_date)]
            
            # Sort by timestamp
            combined_df = combined_df.sort_values('Timestamp')
            
            # Generate output filename
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f'attendance_report_{timestamp}.xlsx'
            
            output_path = os.path.join(self.log_dir, output_file)
            
            # Export to Excel with formatting
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main sheet with all records
                combined_df.to_excel(writer, sheet_name='All Records', index=False)
                
                # Summary sheet - daily attendance count
                daily_summary = combined_df.groupby(combined_df['Timestamp'].dt.date)['Name'].nunique()
                daily_summary_df = pd.DataFrame({
                    'Date': daily_summary.index,
                    'Unique Attendees': daily_summary.values
                })
                daily_summary_df.to_excel(writer, sheet_name='Daily Summary', index=False)
                
                # Person summary - total attendance per person
                person_summary = combined_df.groupby('Name').agg({
                    'Timestamp': 'count',
                    'Confidence': 'mean'
                }).rename(columns={'Timestamp': 'Total Entries', 'Confidence': 'Avg Confidence'})
                person_summary.to_excel(writer, sheet_name='Person Summary')
            
            return output_path
            
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            return None
    
    def get_statistics(self, date=None):
        """
        Get attendance statistics for a specific date.
        
        Args:
            date (str, optional): Date in 'YYYY-MM-DD' format. Defaults to today.
        
        Returns:
            dict: Dictionary containing statistics
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        attendance_file = os.path.join(self.log_dir, f'attendance_{date}.csv')
        
        if not os.path.exists(attendance_file):
            return {
                'date': date,
                'total_entries': 0,
                'unique_attendees': 0,
                'attendee_list': []
            }
        
        attendance = []
        with open(attendance_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                attendance.append(row)
        
        unique_names = set(record['Name'] for record in attendance)
        
        return {
            'date': date,
            'total_entries': len(attendance),
            'unique_attendees': len(unique_names),
            'attendee_list': sorted(list(unique_names))
        }
    
    def clear_recent_recognitions(self):
        """Clear the recent recognitions cache."""
        self.recent_recognitions.clear()
    
    def set_duplicate_threshold(self, seconds):
        """
        Set the duplicate detection threshold.
        
        Args:
            seconds (int): Number of seconds to consider as duplicate
        """
        self.duplicate_threshold = seconds


# Example usage
if __name__ == "__main__":
    logger = AttendanceLogger()
    
    # Log some attendance
    logger.log_attendance("Pratham", 0.95)
    logger.log_attendance("John Doe", 0.87)
    
    # Get today's attendance
    print("Today's Attendance:")
    for record in logger.get_todays_attendance():
        print(f"  {record['Timestamp']} - {record['Name']} ({record['Confidence']})")
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Unique attendees: {stats['unique_attendees']}")
    print(f"  Attendees: {', '.join(stats['attendee_list'])}")
