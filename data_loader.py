import pandas as pd
import glob
import os
import streamlit as st

class DataLoader:
    def __init__(self, data_dir='data/input'):
        self.data_dir = data_dir

    def get_files(self, pattern):
        """Finds files matching a glob pattern in the data directory."""
        full_pattern = os.path.join(self.data_dir, pattern)
        return glob.glob(full_pattern)

    def _read_csv(self, file_path_or_buffer):
        """Helper to read CSV with encoding fallback."""
        encodings = ['utf-8', 'cp1252', 'latin1']
        for enc in encodings:
            try:
                # If it's a buffer (uploaded file), we need to seek 0 if we retry
                if hasattr(file_path_or_buffer, 'seek'):
                    file_path_or_buffer.seek(0)
                return pd.read_csv(file_path_or_buffer, encoding=enc)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # If it's not an encoding error, raise it
                raise e
        # If all failed
        raise UnicodeDecodeError(f"Failed to read CSV with encodings: {encodings}")

    def load_incidents(self, uploaded_file=None):
        """
        Loads Incidents from a CSV file. 
        Prioritizes uploaded_file if provided, otherwise looks for *IM*.csv in data_dir.
        """
        df = pd.DataFrame()
        
        target_file = None
        if uploaded_file:
            target_file = uploaded_file
        else:
            files = self.get_files('*IM*.csv')
            if not files:
                st.warning("No Incident files found in data directory (pattern: *IM*.csv).")
                return df
            target_file = files[0]
            st.success(f"Loaded Incidents from: {os.path.basename(target_file)}")

        try:
            df = self._read_csv(target_file)
        except Exception as e:
            st.error(f"Error reading incidents file: {e}")
            return df

        # Cleaning and Transformation
        # Expected columns based on inspection: number, sys_created_on, short_description, opened_at, closed_at
        
        # Date conversion
        date_cols = ['opened_at', 'closed_at', 'sys_created_on', 'u_resolved']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Fill N/As and Ensure Columns Exist
        text_cols = ['short_description', 'description', 'assignment_group', 'close_code', 'close_notes', 'state', 'incident_state']
        for col in text_cols:
            if col not in df.columns:
                df[col] = ''
            else:
                df[col] = df[col].fillna('')

        # Ensure consistent state column
        if 'incident_state' in df.columns and 'state' not in df.columns:
            df['state'] = df['incident_state']
        
        return df

    def load_changes(self, uploaded_file=None):
        """
        Loads Changes from a CSV file.
        Pattern: *CHANGES*.csv
        """
        df = pd.DataFrame()
        
        target_file = None
        if uploaded_file:
            target_file = uploaded_file
        else:
            files = self.get_files('*CHANGES*.csv')
            if not files:
                st.warning("No Change files found in data directory (pattern: *CHANGES*.csv).")
                return df
            target_file = files[0]
            st.success(f"Loaded Changes from: {os.path.basename(files[0])}")

        try:
            df = self._read_csv(target_file)
        except Exception as e:
             st.error(f"Error reading changes file: {e}")
             return df

        # Cleaning
        # Date conversion
        date_cols = ['start_date', 'end_date', 'closed_at', 'sys_created_on']
        for col in date_cols:
            if col in df.columns:
                # Changes dates often have different formats, but pandas is usually smart enough
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Ensure closed_at exists (Critical for analysis)
        if 'closed_at' not in df.columns:
            # Fallback strategy
            metrics = ['end_date', 'work_end', 'u_resolved', 'start_date']
            for m in metrics:
                if m in df.columns:
                    st.warning(f"'closed_at' missing. Using '{m}' as proxy.")
                    df['closed_at'] = df[m]
                    break
            
            # If still missing, create empty datetime column to prevent KeyErrors
            if 'closed_at' not in df.columns:
                st.error("'closed_at' column missing and no suitable proxy found in Changes CSV.")
                df['closed_at'] = pd.NaT
        
        # Fill N/As and match schema
        text_cols = ['short_description', 'description', 'assignment_group', 'state', 'close_code']
        for col in text_cols:
            if col not in df.columns:
                df[col] = ''
            else:
                df[col] = df[col].fillna('')

        return df

    def load_problems(self, uploaded_file=None):
        """
        Loads Problems from a CSV file.
        Pattern: *PM*.csv or *RCA*.csv? The user mentioned problems (2). 
        Files seen: 'PYTHON EMEA PM P1P2 (This Year).csv'
        """
        df = pd.DataFrame()
        
        target_file = None
        if uploaded_file:
             target_file = uploaded_file
        else:
            # Look for PM files
            files = self.get_files('*PM*.csv')
            if not files:
                 st.warning("No Problem files found in data directory (pattern: *PM*.csv).")
                 return df
            target_file = files[0]
            st.success(f"Loaded Problems from: {os.path.basename(files[0])}")

        try:
            df = self._read_csv(target_file)
        except Exception as e:
            st.error(f"Error reading problems file: {e}")
            return df

        # Cleaning
        # Date conversion - Added u_resolved as per inspection
        date_cols = ['opened_at', 'closed_at', 'sys_created_on', 'u_resolved']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # If u_resolved is present but closed_at is missing (common in some problem exports), fill closed_at
        if 'u_resolved' in df.columns and 'closed_at' in df.columns:
             df['closed_at'] = df['closed_at'].fillna(df['u_resolved'])
        elif 'u_resolved' in df.columns and 'closed_at' not in df.columns:
             df['closed_at'] = df['u_resolved']

        text_cols = ['short_description', 'description', 'assignment_group', 'state', 'root_cause', 'problem_id', 'u_ci_type']
        for col in text_cols:
            if col not in df.columns:
                df[col] = ''
            else:
                df[col] = df[col].fillna('')
                
        return df
